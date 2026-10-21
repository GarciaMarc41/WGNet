"""Differentiable all-Gaussian normalized softmax renderer and dense geological losses in PyTorch."""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

FIELD_SIDE = 70
EDGE_LOSS_WEIGHT = 0.3
DENSITY_LOSS_WEIGHT = 0.1


def make_pixel_grid(
    height: int = FIELD_SIDE,
    width: int = FIELD_SIDE,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return `[height * width, 2]` pixel-center coordinates `(x + 0.5, z + 0.5)` in row-major order."""
    zs = torch.arange(height, device=device, dtype=dtype) + 0.5
    xs = torch.arange(width, device=device, dtype=dtype) + 0.5
    grid_z, grid_x = torch.meshgrid(zs, xs, indexing="ij")
    return torch.stack([grid_x.reshape(-1), grid_z.reshape(-1)], dim=-1)


def render_gaussians(
    params: torch.Tensor,
    mask: torch.Tensor | None = None,
    height: int = FIELD_SIDE,
    width: int = FIELD_SIDE,
    pixel_chunk: int = 4900,
) -> torch.Tensor:
    """Render batch of anisotropic 2D Gaussians via normalized softmax blending.

    Args:
        params: `[B, N, 6]` or `[N, 6]` tensor of `(x, z, angle, log_sx, log_sz, velocity_km)`.
        mask: Optional `[B, N]` or `[N]` additive logit mask (`0.0` for active Gaussians, `-1e30` for padded).
        height: Field height in native pixels (default 70).
        width: Field width in native pixels (default 70).
        pixel_chunk: Query chunk size along pixels to bound peak memory on large batches.

    Returns:
        Rendered velocity field `[B, height, width]` (or `[height, width]` if unbatched) in km/s.
    """
    unbatched = params.ndim == 2
    if unbatched:
        params = params.unsqueeze(0)
        if mask is not None and mask.ndim == 1:
            mask = mask.unsqueeze(0)

    bsz, n_gauss, ch = params.shape
    assert ch == 6, f"Expected 6 parameters per Gaussian, got {ch}"

    xy = make_pixel_grid(height, width, device=params.device, dtype=params.dtype)  # [P, 2]
    num_pixels = xy.shape[0]

    # Split Gaussian attributes: each [B, 1, N]
    gx = params[:, :, 0].unsqueeze(1)
    gz = params[:, :, 1].unsqueeze(1)
    angle = params[:, :, 2].unsqueeze(1)
    log_sx = params[:, :, 3].unsqueeze(1)
    log_sz = params[:, :, 4].unsqueeze(1)
    val = params[:, :, 5]  # [B, N]

    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    inv_var_x = torch.exp(-2.0 * log_sx)
    inv_var_z = torch.exp(-2.0 * log_sz)

    chunks = []
    for start in range(0, num_pixels, pixel_chunk):
        end = min(num_pixels, start + pixel_chunk)
        q_xy = xy[start:end]  # [Q, 2]
        dx = q_xy[:, 0].view(1, -1, 1) - gx  # [B, Q, N]
        dz = q_xy[:, 1].view(1, -1, 1) - gz  # [B, Q, N]

        u = dx * cos_a + dz * sin_a
        v = dz * cos_a - dx * sin_a
        logits = -0.5 * (u.square() * inv_var_x + v.square() * inv_var_z)  # [B, Q, N]
        if mask is not None:
            logits = logits + mask.unsqueeze(1)

        weights = F.softmax(logits, dim=-1)  # [B, Q, N]
        pred_chunk = torch.einsum("bqn,bn->bq", weights, val)  # [B, Q]
        chunks.append(pred_chunk)

    field = torch.cat(chunks, dim=1).view(bsz, height, width)
    return field.squeeze(0) if unbatched else field


def field_and_edge_loss(
    pred_km: torch.Tensor,
    target_km: torch.Tensor,
    edge_weight: float = EDGE_LOSS_WEIGHT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute exact dense field MSE and direction-balanced adjacent-difference edge MSE in (km/s)^2.

    Args:
        pred_km: `[B, H, W]` predicted velocity in km/s.
        target_km: `[B, H, W]` ground-truth velocity in km/s.
        edge_weight: Weight on adjacent-difference MSE (default 0.3).

    Returns:
        `(total_geo_loss, field_mse, edge_mse)`
    """
    err = pred_km - target_km
    field_mse = err.square().mean()

    dx_err = err[:, :, 1:] - err[:, :, :-1]
    dz_err = err[:, 1:, :] - err[:, :-1, :]
    edge_mse = 0.5 * (dx_err.square().mean() + dz_err.square().mean())

    total = field_mse + edge_weight * edge_mse
    return total, field_mse, edge_mse


def density_supervision_loss(
    logits: torch.Tensor,
    density: torch.Tensor,
    teacher_density: torch.Tensor,
) -> torch.Tensor:
    """Stable sigmoid cross-entropy + total occupancy mass calibration penalty.
    Matches `src/direct/network.ts` lines 257-259.
    """
    cross_entropy = (F.softplus(logits) - logits * teacher_density).mean()
    mass_penalty = (density.mean(dim=(-2, -1)) - teacher_density.mean(dim=(-2, -1))).square().mean()
    return cross_entropy + mass_penalty


@torch.no_grad()
def compute_metrics_mps(pred_km: torch.Tensor, target_km: torch.Tensor) -> dict[str, float]:
    """Compute pooled field RMSE (m/s) and adjacent-cell gradient RMSE (m/s) matching `analyze-direct.py`."""
    err_mps = (pred_km - target_km) * 1000.0
    field_rmse = float(torch.sqrt(err_mps.square().mean()).item())

    dx = err_mps[..., :, 1:] - err_mps[..., :, :-1]
    dz = err_mps[..., 1:, :] - err_mps[..., :-1, :]
    all_edges = torch.cat([dx.reshape(-1), dz.reshape(-1)], dim=0)
    grad_rmse = float(torch.sqrt(all_edges.square().mean()).item())
    return {"field_rmse_mps": field_rmse, "grad_rmse_mps": grad_rmse}
