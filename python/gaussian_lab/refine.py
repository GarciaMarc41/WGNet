"""Matched-budget wave-only L-BFGS refinement for Gaussian and Grid representations.

Mirrors the exact refinement protocol of `src/wavefield/refinement.ts`:
- True geology is never passed to the optimizer.
- Uses the exact discrete `wgpu` wave forward/adjoint solver (`storeEvery = 1`).
- Tracks every forward and adjoint survey solve (including rejected line-search candidates)
  up to the 160-survey-solve budget.
"""

from __future__ import annotations
import math
import time
from typing import NamedTuple
import numpy as np
import torch
import wave_gauss_rs

from .data import DT, H, N, NT, PEAK_HZ, SHOTS
from .renderer import FIELD_SIDE, render_gaussians

GAUSSIAN_SCALES = np.array([2.0, 2.0, 0.15, 0.15, 0.15, 0.15], dtype=np.float64)
GRID_SCALE = 0.15
POS_BOUNDS = (-3.5, 73.5)
LOG_SIGMA_BOUNDS = (math.log(0.05), math.log(140.0))
VEL_BOUNDS_KM = (1.0, 5.0)
LBFGS_PAIRS = 6
LINE_SEARCH_TRIALS = 6
ARMIJO = 1e-4
WAVE_TOLERANCE = 1e-6


class RefinementStepRecord(NamedTuple):
    cost: float
    iteration: int
    loss: float
    seconds: float
    event: str
    count: int
    forward_solves: int
    adjoint_solves: int


class RefinementResult(NamedTuple):
    initial_field_km: np.ndarray
    final_field_km: np.ndarray
    initial_loss: float
    final_loss: float
    actual_cost: float
    forward_solves: int
    adjoint_solves: int
    seconds: float
    stop_reason: str
    history: list[RefinementStepRecord]
    final_params: np.ndarray


def _project_gaussian_params(params: np.ndarray) -> np.ndarray:
    out = params.copy().reshape(-1, 6)
    out[:, 0] = np.clip(out[:, 0], POS_BOUNDS[0], POS_BOUNDS[1])
    out[:, 1] = np.clip(out[:, 1], POS_BOUNDS[0], POS_BOUNDS[1])
    out[:, 3] = np.clip(out[:, 3], LOG_SIGMA_BOUNDS[0], LOG_SIGMA_BOUNDS[1])
    out[:, 4] = np.clip(out[:, 4], LOG_SIGMA_BOUNDS[0], LOG_SIGMA_BOUNDS[1])
    out[:, 5] = np.clip(out[:, 5], VEL_BOUNDS_KM[0], VEL_BOUNDS_KM[1])
    return out.reshape(-1)


def _lbfgs_direction(
    grad: np.ndarray,
    history_s: list[np.ndarray],
    history_y: list[np.ndarray],
    history_rho: list[float],
) -> np.ndarray:
    q = grad.copy()
    m = len(history_s)
    alphas = [0.0] * m
    for i in range(m - 1, -1, -1):
        alphas[i] = history_rho[i] * float(np.dot(history_s[i], q))
        q -= alphas[i] * history_y[i]
    if m > 0:
        sy = float(np.dot(history_s[-1], history_y[-1]))
        yy = float(np.dot(history_y[-1], history_y[-1]))
        gamma = sy / max(yy, 1e-30) if sy > 0.0 and yy > 0.0 else 1.0
        q *= gamma
    for i in range(m):
        beta = history_rho[i] * float(np.dot(history_y[i], q))
        q += history_s[i] * (alphas[i] - beta)
    return -q


def refine_gaussians_with_waves(
    initial_params: np.ndarray,
    observed_traces: np.ndarray,
    engine: wave_gauss_rs.PyWaveEngine | None = None,
    budget_solves: int = 160,
    device: torch.device | str = "cpu",
) -> RefinementResult:
    """Refine `[N, 6]` Gaussian parameters against observed wave recordings under a fixed solve budget."""
    t0 = time.perf_counter()
    if engine is None:
        engine = wave_gauss_rs.PyWaveEngine(SHOTS, NT, DT, PEAK_HZ, 20, 1000.0, 5000.0)

    obs_flat = np.ascontiguousarray(observed_traces.reshape(-1), dtype=np.float32)
    obs_energy = float(np.square(obs_flat, dtype=np.float64).sum())

    cur_raw = _project_gaussian_params(np.asarray(initial_params, dtype=np.float64).reshape(-1))
    count = len(cur_raw) // 6
    scales = np.tile(GAUSSIAN_SCALES, count)

    forward_solves = 0
    adjoint_solves = 0

    def render_field_and_vjp(raw_vec: np.ndarray, field_grad_mps: np.ndarray | None = None):
        p_t = torch.tensor(
            raw_vec.reshape(count, 6), dtype=torch.float32, device=device, requires_grad=(field_grad_mps is not None)
        )
        field_t = render_gaussians(p_t, height=FIELD_SIDE, width=FIELD_SIDE)
        field_np = field_t.detach().cpu().numpy().astype(np.float32)
        if field_grad_mps is None:
            return field_np, None
        # Chain dJ/d(v in m/s) -> dJ/d(v in km/s) = 1000 * field_grad_mps
        g_t = torch.from_numpy(field_grad_mps * 1000.0).to(device=device, dtype=torch.float32)
        (param_grad_t,) = torch.autograd.grad((field_t * g_t).sum(), p_t)
        return field_np, param_grad_t.detach().cpu().numpy().astype(np.float64).reshape(-1)

    def eval_candidate(raw_vec: np.ndarray, need_grad: bool):
        nonlocal forward_solves, adjoint_solves
        field_km, _ = render_field_and_vjp(raw_vec, None)
        vel_mps = np.ascontiguousarray((field_km.reshape(-1) * 1000.0), dtype=np.float32)
        if need_grad:
            forward_solves += 1
            adjoint_solves += 1
            loss, grad_mps = engine.evaluate_adjoint(vel_mps, obs_flat)
            _, param_grad = render_field_and_vjp(raw_vec, grad_mps)
            return float(loss), field_km, param_grad * scales
        else:
            forward_solves += 1
            pred_traces = engine.simulate(vel_mps).reshape(-1)
            diff = pred_traces.astype(np.float64) - obs_flat.astype(np.float64)
            loss = 0.5 * float(np.dot(diff, diff)) / obs_energy
            return loss, field_km, None

    cur_loss, initial_field_km, cur_grad = eval_candidate(cur_raw, need_grad=True)
    cur_field_km = initial_field_km.copy()
    initial_loss = cur_loss

    history = [
        RefinementStepRecord(
            cost=float(forward_solves + adjoint_solves),
            iteration=0,
            loss=cur_loss,
            seconds=time.perf_counter() - t0,
            event="initial",
            count=count,
            forward_solves=forward_solves,
            adjoint_solves=adjoint_solves,
        )
    ]

    hist_s: list[np.ndarray] = []
    hist_y: list[np.ndarray] = []
    hist_rho: list[float] = []
    iteration = 0
    stop_reason = "budget"

    while (forward_solves + adjoint_solves) < budget_solves:
        if cur_loss <= WAVE_TOLERANCE:
            stop_reason = "tolerance"
            break

        direction = _lbfgs_direction(cur_grad, hist_s, hist_y, hist_rho)
        slope = float(np.dot(cur_grad, direction))
        if not np.isfinite(slope) or slope >= 0.0:
            hist_s.clear()
            hist_y.clear()
            hist_rho.clear()
            direction = -cur_grad
            slope = -float(np.dot(cur_grad, cur_grad))

        d_norm = float(np.linalg.norm(direction))
        if d_norm > 1.0:
            direction /= d_norm
            slope /= d_norm

        accepted = False
        for pass_idx in range(2):
            step = 1.0
            for _ in range(LINE_SEARCH_TRIALS):
                if (forward_solves + adjoint_solves) >= budget_solves:
                    break
                cand_raw = _project_gaussian_params(cur_raw + step * scales * direction)
                cand_loss, cand_field, _ = eval_candidate(cand_raw, need_grad=False)
                if cand_loss <= cur_loss + ARMIJO * step * slope:
                    if (forward_solves + adjoint_solves) + 2 <= budget_solves:
                        cand_loss, cand_field, cand_grad = eval_candidate(cand_raw, need_grad=True)
                        s_vec = (cand_raw - cur_raw) / scales
                        y_vec = cand_grad - cur_grad
                        sy = float(np.dot(s_vec, y_vec))
                        if sy > 1e-12:
                            if len(hist_s) >= LBFGS_PAIRS:
                                hist_s.pop(0)
                                hist_y.pop(0)
                                hist_rho.pop(0)
                            hist_s.append(s_vec)
                            hist_y.append(y_vec)
                            hist_rho.append(1.0 / sy)
                        cur_raw = cand_raw
                        cur_loss = cand_loss
                        cur_field_km = cand_field
                        cur_grad = cand_grad
                    else:
                        cur_raw = cand_raw
                        cur_loss = cand_loss
                        cur_field_km = cand_field
                    accepted = True
                    iteration += 1
                    history.append(
                        RefinementStepRecord(
                            cost=float(forward_solves + adjoint_solves),
                            iteration=iteration,
                            loss=cur_loss,
                            seconds=time.perf_counter() - t0,
                            event="accepted",
                            count=count,
                            forward_solves=forward_solves,
                            adjoint_solves=adjoint_solves,
                        )
                    )
                    break
                step *= 0.5
            if accepted or (forward_solves + adjoint_solves) >= budget_solves:
                break
            if pass_idx == 0 and len(hist_s) > 0:
                hist_s.clear()
                hist_y.clear()
                hist_rho.clear()
                direction = -cur_grad
                d_norm = float(np.linalg.norm(direction))
                if d_norm > 1.0:
                    direction /= d_norm
                slope = float(np.dot(cur_grad, direction))
            else:
                break

        if not accepted:
            stop_reason = "budget" if (forward_solves + adjoint_solves) >= budget_solves else "stalled"
            break

    return RefinementResult(
        initial_field_km=initial_field_km,
        final_field_km=cur_field_km,
        initial_loss=initial_loss,
        final_loss=cur_loss,
        actual_cost=float(forward_solves + adjoint_solves),
        forward_solves=forward_solves,
        adjoint_solves=adjoint_solves,
        seconds=time.perf_counter() - t0,
        stop_reason=stop_reason,
        history=history,
        final_params=cur_raw.reshape(count, 6).astype(np.float32),
    )
