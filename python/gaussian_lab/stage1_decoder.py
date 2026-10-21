"""Stage 1: Self-Contained Geological Autoencoder & Triangle-Conditioned Gaussian Decoder.

Learns a strong geological Gaussian decoder `D_gauss(z_coarse, z_fine) -> (density, Gaussians, v_hat)`
before wave recordings are introduced. Never accesses target velocities in the Gaussian head;
velocity residuals are predicted purely from latent spatial features around the training mean field.
"""

from __future__ import annotations
import math
from typing import NamedTuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wave_gauss_rs

from .renderer import FIELD_SIDE, render_gaussians


class DecoderOutput(NamedTuple):
    field_km: torch.Tensor       # [B, 70, 70] rendered velocity in km/s
    logits: torch.Tensor         # [B, 70, 70] PPM logits
    density: torch.Tensor        # [B, 70, 70] PPM probabilities in (0, 1)
    params: torch.Tensor         # [B, N_pad, 6] Gaussian attributes
    mask: torch.Tensor           # [B, N_pad] additive logit mask (0 or -1e30)
    counts: torch.Tensor         # [B] active triangle/Gaussian count per example
    z_coarse: torch.Tensor       # [B, 64, 35, 35] coarse latent features
    z_fine: torch.Tensor         # [B, 48, 70, 70] fine latent features


class ResidualBlock2d(nn.Module):
    def __init__(self, channels: int, groups: int = 8) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode="replicate")
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode="replicate")
        self.norm2 = nn.GroupNorm(groups, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.silu(x + 0.5 * h)


class GeologyLatentEncoder(nn.Module):
    """Encodes a 70x70 velocity field (km/s) into normalized multi-scale spatial latents `(z_coarse, z_fine)`."""

    def __init__(self, coarse_ch: int = 64, fine_ch: int = 48) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, fine_ch, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.GroupNorm(8, fine_ch),
            nn.SiLU(),
            ResidualBlock2d(fine_ch, groups=8),
            ResidualBlock2d(fine_ch, groups=8),
        )
        self.down = nn.Sequential(
            nn.Conv2d(fine_ch, coarse_ch, kernel_size=3, stride=2, padding=1, padding_mode="replicate"),
            nn.GroupNorm(8, coarse_ch),
            nn.SiLU(),
            ResidualBlock2d(coarse_ch, groups=8),
            ResidualBlock2d(coarse_ch, groups=8),
        )
        self.coarse_norm = nn.GroupNorm(8, coarse_ch)
        self.fine_norm = nn.GroupNorm(8, fine_ch)

    def forward(self, v_km: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if v_km.ndim == 3:
            v_km = v_km.unsqueeze(1)
        x = (v_km - 3.0) * 0.5  # [-1, 1]
        fine_raw = self.stem(x)  # [B, 48, 70, 70]
        coarse_raw = self.down(fine_raw)  # [B, 64, 35, 35]
        z_fine = self.fine_norm(fine_raw)
        z_coarse = self.coarse_norm(coarse_raw)
        return z_coarse, z_fine


class TriangleGaussianDecoder(nn.Module):
    """Maps spatial latents `(z_coarse [B, 64, 35, 35], z_fine [B, 48, 70, 70])` to adaptive Delaunay Gaussians."""

    def __init__(
        self,
        mean_km: np.ndarray | torch.Tensor,
        coarse_ch: int = 64,
        fine_ch: int = 48,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        mean_t = torch.as_tensor(mean_km, dtype=torch.float32).view(1, 1, FIELD_SIDE, FIELD_SIDE)
        base_atanh = torch.atanh(torch.clamp((mean_t - 3.0) * 0.5, -0.999, 0.999))
        self.register_buffer("base_atanh", base_atanh)

        # Fuse coarse + fine latents into refined 70x70 detail features
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(coarse_ch + fine_ch, fine_ch, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.GroupNorm(8, fine_ch),
            nn.SiLU(),
            ResidualBlock2d(fine_ch, groups=8),
        )

        # Position Probability Map (PPM) head
        self.ppm_head = nn.Conv2d(fine_ch, 1, kernel_size=3, padding=1, padding_mode="replicate")
        nn.init.normal_(self.ppm_head.weight, mean=0.0, std=0.001)
        nn.init.constant_(self.ppm_head.bias, math.log(0.05 / 0.95))

        # 4 sample points (3 triangle vertices + 1 centroid) x (fine_ch + coarse_ch) + 5 Steiner priors
        in_features = 4 * (fine_ch + coarse_ch) + 5
        self.attr_mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 7),
        )
        nn.init.normal_(self.attr_mlp[-1].weight, mean=0.0, std=0.002)
        nn.init.zeros_(self.attr_mlp[-1].bias)

    @staticmethod
    def _to_grid_sample_coords(coords_px: torch.Tensor, size: int = FIELD_SIDE) -> torch.Tensor:
        """Convert pixel-edge coordinates `[0, size]` (where pixel centers are `i + 0.5`)
        to `F.grid_sample(..., align_corners=True)` normalized coordinates `[-1, 1]`.
        """
        return (coords_px - 0.5) / ((size - 1) * 0.5) - 1.0

    def forward(
        self,
        z_coarse: torch.Tensor,
        z_fine: torch.Tensor,
        density_override: torch.Tensor | None = None,
    ) -> DecoderOutput:
        bsz = z_fine.shape[0]
        device = z_fine.device

        up_coarse = F.interpolate(z_coarse, size=(FIELD_SIDE, FIELD_SIDE), mode="bilinear", align_corners=False)
        detail = self.fuse_conv(torch.cat([z_fine, up_coarse], dim=1))  # [B, 48, 70, 70]

        logits = self.ppm_head(detail).squeeze(1)  # [B, 70, 70]
        density = torch.sigmoid(logits)

        # Detached discretization & Delaunay triangulation in Rust (Rayon parallel across batch)
        ppm_for_layout = density_override if density_override is not None else density
        ppm_np = np.ascontiguousarray(
            ppm_for_layout.detach().cpu().numpy().reshape(bsz, FIELD_SIDE * FIELD_SIDE),
            dtype=np.float32,
        )
        verts_np, priors_np, sample_np, mask_np, counts_np = wave_gauss_rs.batch_density_to_layout(
            ppm_np, FIELD_SIDE, FIELD_SIDE, 64
        )
        n_pad = verts_np.shape[1]

        vertices = torch.from_numpy(verts_np).to(device=device).view(bsz, n_pad, 3, 2)
        priors = torch.from_numpy(priors_np).to(device=device)          # [B, N_pad, 5]
        sample_coords = torch.from_numpy(sample_np).to(device=device).view(bsz, n_pad, 4, 2)
        mask = torch.from_numpy(mask_np).to(device=device)              # [B, N_pad]
        counts = torch.from_numpy(counts_np.astype(np.int64)).to(device=device)

        # Sub-pixel bilinear feature sampling at the 3 triangle vertices + centroid
        grid_4 = self._to_grid_sample_coords(sample_coords, FIELD_SIDE)  # [B, N_pad, 4, 2]
        fine_samples = F.grid_sample(
            detail, grid_4, mode="bilinear", padding_mode="border", align_corners=True
        )  # [B, 48, N_pad, 4]
        coarse_samples = F.grid_sample(
            up_coarse, grid_4, mode="bilinear", padding_mode="border", align_corners=True
        )  # [B, 64, N_pad, 4]

        sampled_feat = (
            torch.cat([fine_samples, coarse_samples], dim=1)
            .permute(0, 2, 3, 1)
            .reshape(bsz, n_pad, -1)
        )  # [B, N_pad, 4 * (48 + 64)]

        # Sample immutable training-mean base at triangle centroid
        centroid_grid = grid_4[:, :, 3:4, :]  # [B, N_pad, 1, 2]
        base_atanh = F.grid_sample(
            self.base_atanh.expand(bsz, -1, -1, -1),
            centroid_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).view(bsz, n_pad, 1)

        # Normalize Steiner geometry priors matching `src/direct/network.ts`
        prior_scale = torch.tensor([35.0, 35.0, math.pi, 4.0, 4.0], device=device)
        prior_shift = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0], device=device)
        geom_feat = priors / prior_scale - prior_shift

        mlp_in = torch.cat([sampled_feat, geom_feat], dim=-1)
        raw = self.attr_mlp(mlp_in)  # [B, N_pad, 7]

        bary = F.softmax(raw[:, :, 0:3], dim=-1).unsqueeze(-1)  # [B, N_pad, 3, 1]
        position = (bary * vertices).sum(dim=2)                 # [B, N_pad, 2]

        log_widths = torch.clamp(
            priors[:, :, 3:5] + torch.tanh(raw[:, :, 3:5]) * 1.5,
            math.log(0.05),
            math.log(140.0),
        )
        angle = priors[:, :, 2:3] + torch.tanh(raw[:, :, 5:6]) * (math.pi * 0.5)
        velocity = 3.0 + 2.0 * torch.tanh(raw[:, :, 6:7] + base_atanh)

        params = torch.cat([position, angle, log_widths, velocity], dim=-1)  # [B, N_pad, 6]
        field_km = render_gaussians(params, mask=mask, height=FIELD_SIDE, width=FIELD_SIDE)

        return DecoderOutput(
            field_km=field_km,
            logits=logits,
            density=density,
            params=params,
            mask=mask,
            counts=counts,
            z_coarse=z_coarse,
            z_fine=z_fine,
        )


class GeologicalAutoencoder(nn.Module):
    """Complete Stage 1 model: `v_true -> E_geo -> (z_coarse, z_fine) -> D_gauss -> v_hat`."""

    def __init__(
        self,
        mean_km: np.ndarray | torch.Tensor,
        coarse_ch: int = 64,
        fine_ch: int = 48,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = GeologyLatentEncoder(coarse_ch=coarse_ch, fine_ch=fine_ch)
        self.decoder = TriangleGaussianDecoder(
            mean_km=mean_km, coarse_ch=coarse_ch, fine_ch=fine_ch, hidden_dim=hidden_dim
        )

    def forward(
        self,
        v_km: torch.Tensor,
        density_override: torch.Tensor | None = None,
    ) -> DecoderOutput:
        z_coarse, z_fine = self.encoder(v_km)
        return self.decoder(z_coarse, z_fine, density_override=density_override)
