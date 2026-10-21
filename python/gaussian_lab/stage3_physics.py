"""Stage 3: Differentiable PyTorch Autograd Bridge to the Native Rust `wgpu` Wave Solver.

Provides exact discrete wave-equation forward and adjoint gradients (`splatForward.wgsl` + `splatAdjoint.wgsl`)
as a first-class PyTorch loss `WavePhysicsLoss(pred_field_km, observed_traces)` that backpropagates
through the differentiable Gaussian renderer into both the Gaussian Decoder and Wave Encoder.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import wave_gauss_rs

from .data import DT, H, N, NT, PEAK_HZ, SHOTS


class WavePhysicsFunction(torch.autograd.Function):
    """Custom `torch.autograd.Function` chaining `PyWaveEngine.evaluate_adjoint` with PyTorch autograd."""

    @staticmethod
    def forward(
        ctx,
        field_km: torch.Tensor,
        observed_traces: torch.Tensor,
        engine: wave_gauss_rs.PyWaveEngine,
        single_shot_engine: wave_gauss_rs.PyWaveEngine | None = None,
        shot_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            field_km: `[B, 70, 70]` predicted velocity fields in km/s.
            observed_traces: `[B, 5, 2001, 70]` observed shot gathers.
            engine: 5-shot `PyWaveEngine` instance.
            single_shot_engine: Optional 1-shot `PyWaveEngine` for fast stochastic single-shot adjoint.
            shot_indices: Optional `[B]` shot index in `0..4` per batch item when using `single_shot_engine`.
        """
        bsz = field_km.shape[0]
        device = field_km.device
        dtype = field_km.dtype

        fields_mps_np = np.ascontiguousarray(
            (field_km.detach().cpu().numpy() * 1000.0).reshape(bsz, N * N),
            dtype=np.float32,
        )
        obs_np = np.ascontiguousarray(
            observed_traces.detach().cpu().numpy(),
            dtype=np.float32,
        )

        grads_km = np.zeros((bsz, N, N), dtype=np.float32)
        losses = np.zeros(bsz, dtype=np.float32)
        xmax = (N - 1) * H

        for b in range(bsz):
            if single_shot_engine is not None and shot_indices is not None:
                s_idx = int(shot_indices[b].item())
                src_x = xmax * 0.5 if SHOTS == 1 else (xmax * s_idx) / (SHOTS - 1)
                src_arr = np.array([[src_x, H]], dtype=np.float32)
                single_shot_engine.set_sources(src_arr)
                obs_1 = np.ascontiguousarray(obs_np[b, s_idx].reshape(-1), dtype=np.float32)
                loss_b, grad_mps_b = single_shot_engine.evaluate_adjoint(fields_mps_np[b], obs_1)
            else:
                obs_all = np.ascontiguousarray(obs_np[b].reshape(-1), dtype=np.float32)
                loss_b, grad_mps_b = engine.evaluate_adjoint(fields_mps_np[b], obs_all)

            losses[b] = loss_b
            # Chain rule: dJ / d(v_km) = 1000.0 * dJ / d(v_mps)
            grads_km[b] = (grad_mps_b * 1000.0) / bsz

        ctx.save_for_backward(torch.from_numpy(grads_km).to(device=device, dtype=dtype))
        return torch.tensor(float(losses.mean()), device=device, dtype=dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (grads_km,) = ctx.saved_tensors
        return grad_output * grads_km, None, None, None, None


class WavePhysicsLoss(nn.Module):
    """Module wrapper for online wave-equation simulator supervision during Stage 3 training."""

    def __init__(self, use_stochastic_single_shot: bool = True) -> None:
        super().__init__()
        self.full_engine = wave_gauss_rs.PyWaveEngine(SHOTS, NT, DT, PEAK_HZ, 20, 1000.0, 5000.0)
        self.single_shot_engine = (
            wave_gauss_rs.PyWaveEngine(1, NT, DT, PEAK_HZ, 20, 1000.0, 5000.0)
            if use_stochastic_single_shot
            else None
        )
        self.use_stochastic_single_shot = use_stochastic_single_shot

    def forward(
        self,
        field_km: torch.Tensor,
        observed_traces: torch.Tensor,
        full_survey: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if self.use_stochastic_single_shot and not full_survey:
            bsz = field_km.shape[0]
            shot_indices = torch.randint(
                0, SHOTS, (bsz,), generator=generator, device="cpu"
            )
            return WavePhysicsFunction.apply(
                field_km,
                observed_traces,
                self.full_engine,
                self.single_shot_engine,
                shot_indices,
            )
        return WavePhysicsFunction.apply(
            field_km, observed_traces, self.full_engine, None, None
        )
