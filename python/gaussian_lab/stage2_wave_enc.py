"""Stage 2: SOTA Acquisition-Agnostic Multi-Velocity NMO + Axial Depth-Horizon Wave Encoder (`stage2_wave_enc.py`).

Upgrades over v1 (`327.29 m/s`):
1. **3-Channel Wavelet-Preserving Stem (`T' = 63`, `dt' = 16 ms`)**:
   - Channel 0: True-amplitude `asinh(w / rms_scale)` (direct-arrival velocity & shallow kinematics).
   - Channel 1: Time-Gain-Compensated (TGC) `asinh(w * ((t + 0.05)/0.12)^1.4 / (0.12 * rms_scale))` (15x boost for weak late reflections).
   - Channel 2: Temporal derivative `asinh(du/dt / (0.25 * rms_scale))` (90-degree quadrature phase partner that sharpens reflection arrival onsets and polarity inversions `R < 0` on `_B` families).
   - Preserves `T' = 63` (`dt' = 16 ms`) instead of `T' = 32` (`32 ms`), preventing aliasing of the 22 ms half-cycle lobe of the 15 Hz Ricker wavelet.
2. **Multi-Velocity NMO Hyperbola Bank (`M = 4` Trial Velocity Profiles across `1800..4050 m/s`)**:
   - Simultaneously gathers reflection energy along 4 trial RMS velocity hyperbolas so non-monotonic velocity inversions (`FlatVel_B`, `CurveVel_B`, `CurveFault_B`) are focused regardless of overburden speed.
3. **True Identity Pre-Activation Residual Highway (`x + alpha * c2(silu(c1(silu(x))))`)**:
   - Eliminates the outer `F.silu` amplitude attenuation (`0.5^L`) and preserves signed wave polarity across deep layers.
4. **Axial Depth-Horizon Mixer (`Z = 35` Dix Column Integration + `X = 35` Horizon Continuity)**:
   - Performs global vertical Dix traveltime-to-depth stripping along each depth column `z` and lateral reflector continuity along each horizon `x`.
5. **Virtual-Source Gaussian Kernel Basis (`K = 5`)**:
   - Generalizes seamlessly to any number of shots (`S = 1, 2, 3, 5, ...`) at arbitrary source coordinates `source_x_m`.
"""

from __future__ import annotations
import math
from typing import NamedTuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .stage1_decoder import DecoderOutput, GeologyLatentEncoder, TriangleGaussianDecoder


class WaveLatentOutput(NamedTuple):
    z_coarse: torch.Tensor       # [B, 64, 35, 35]
    z_fine: torch.Tensor         # [B, 48, 70, 70]
    v_aux_km: torch.Tensor       # [B, 70, 70] auxiliary direct velocity prediction in km/s
    temporal_attn: torch.Tensor  # [M, S, R', T', Z'] multi-velocity NMO traveltime gather kernels
    spatial_attn: torch.Tensor   # [S, K] virtual-source projection weights


class PureResBlock(nn.Module):
    """True identity pre-activation residual block without spatial GroupNorm (preserves global velocity scale & wave polarity)."""

    def __init__(self, ch: int, dilation: int = 1) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        nn.init.kaiming_normal_(self.c1.weight, nonlinearity="relu")
        nn.init.zeros_(self.c1.bias)
        nn.init.kaiming_normal_(self.c2.weight, nonlinearity="relu")
        nn.init.zeros_(self.c2.bias)
        self.alpha = nn.Parameter(torch.tensor(0.30))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.alpha * self.c2(F.silu(self.c1(F.silu(x))))


class AxialDepthHorizonMixer(nn.Module):
    """Non-local vertical Dix depth-column stripping (`Z=35`) + lateral horizon mixer (`X=35`)."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.z_mix = nn.Linear(35, 35)
        self.x_mix = nn.Linear(35, 35)
        nn.init.eye_(self.z_mix.weight)
        nn.init.zeros_(self.z_mix.bias)
        nn.init.eye_(self.x_mix.weight)
        nn.init.zeros_(self.x_mix.bias)
        self.fuse = nn.Conv2d(ch * 2, ch, kernel_size=1)
        nn.init.kaiming_normal_(self.fuse.weight, nonlinearity="relu")
        nn.init.zeros_(self.fuse.bias)
        self.beta = nn.Parameter(torch.tensor(0.25))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hz = self.z_mix(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        hx = self.x_mix(x)
        return x + self.beta * self.fuse(F.silu(torch.cat([hz, hx], dim=1)))


class HybridVirtualSourceWaveEncoder(nn.Module):
    """SOTA Multi-Velocity NMO + Axial Depth-Horizon Virtual-Source Wave Encoder."""

    def __init__(
        self,
        mean_km: np.ndarray | torch.Tensor,
        mean_bg_traces: np.ndarray | torch.Tensor | None = None,
        rms_scale: float = 1.505,
        scatter_scale: float = 0.903,
        coarse_ch: int = 64,
        fine_ch: int = 48,
        n_modes: int = 5,
    ) -> None:
        super().__init__()
        del mean_bg_traces, scatter_scale  # kept in signature for backwards compatibility
        self.rms_scale = max(float(rms_scale), 1e-4)
        self.n_modes = n_modes
        self.n_vel_bank = 4

        t_norm = torch.linspace(0.0, 1.0, 251).view(1, 1, 251, 1)
        self.register_buffer("tgc_gain", ((t_norm + 0.05) / 0.12).pow(1.4))

        mean_t = torch.as_tensor(mean_km, dtype=torch.float32).view(1, 1, 70, 70)
        base_atanh = torch.atanh(torch.clamp((mean_t - 3.0) * 0.5, -0.999, 0.999))
        self.register_buffer("base_atanh", base_atanh)

        # Multi-Velocity NMO Bank: M=4 velocity profiles spanning [1750 .. 4050] m/s
        v_bank = torch.tensor([1800.0, 2350.0, 3000.0, 3750.0]).view(4, 1) + torch.linspace(
            0.0, 300.0, 35
        ).view(1, 35)
        self.v_bank_mps = nn.Parameter(v_bank)  # [M=4, Z'=35]

        # Wavelet-preserving high-res stem:
        # [B*S, 3, 251, 70] -> [B*S, 32, 126, 70] -> [B*S, 64, 63, 35]
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1)),
            nn.SiLU(),
            PureResBlock(64),
            PureResBlock(64),
        )

        # Branch A: High-resolution learned time-to-depth projection (T'=63 -> Z'=35)
        self.time_proj = nn.Linear(63, 35)
        nn.init.normal_(self.time_proj.weight, 0.0, math.sqrt(1.0 / 63.0))
        nn.init.zeros_(self.time_proj.bias)

        # Branch B: Multi-Velocity NMO Hyperbola Bank fusion (4 * 64 -> 64)
        self.nmo_bank_proj = nn.Conv2d(self.n_vel_bank * 64, 64, kernel_size=1)
        nn.init.kaiming_normal_(self.nmo_bank_proj.weight, nonlinearity="linear")
        nn.init.zeros_(self.nmo_bank_proj.bias)
        self.nmo_scale = nn.Parameter(torch.tensor(0.40))

        # Per-shot offset-aware refinement block before virtual-source projection
        self.shot_refine = nn.Sequential(
            nn.Conv2d(64 + 2, 64, kernel_size=3, padding=1),
            PureResBlock(64),
        )

        self.register_buffer("virtual_xs", torch.linspace(0.0, 690.0, n_modes))

        coords = torch.stack(
            torch.meshgrid(
                torch.linspace(-1.0, 1.0, 35),
                torch.linspace(-1.0, 1.0, 35),
                indexing="ij",
            ),
            dim=0,
        ).unsqueeze(0)
        self.register_buffer("coords", coords)

        # Fuse K virtual modes * 64 + K illumination weights + 2 coords -> 128
        self.fuse = nn.Conv2d(n_modes * 64 + n_modes + 2, 128, kernel_size=1)
        self.fine35 = nn.Sequential(
            PureResBlock(128),
            AxialDepthHorizonMixer(128),
            PureResBlock(128),
            PureResBlock(128, dilation=2),
        )
        self.down18 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.SiLU(),
            PureResBlock(128),
            PureResBlock(128),
        )
        self.join35 = nn.Sequential(
            nn.Conv2d(128 + 128, 96, 3, padding=1),
            nn.SiLU(),
            PureResBlock(96),
            AxialDepthHorizonMixer(96),
            PureResBlock(96),
        )
        self.up70 = nn.Sequential(
            nn.Conv2d(96, fine_ch, 3, padding=1),
            nn.SiLU(),
            PureResBlock(fine_ch),
            PureResBlock(fine_ch),
            nn.Conv2d(fine_ch, fine_ch, 3, padding=1),
            nn.SiLU(),
        )

        self.zc_proj = nn.Sequential(nn.Conv2d(96, coarse_ch, 1), nn.GroupNorm(8, coarse_ch))
        self.zf_proj = nn.Sequential(nn.Conv2d(fine_ch, fine_ch, 1), nn.GroupNorm(8, fine_ch))
        self.v_head = nn.Conv2d(fine_ch, 1, kernel_size=3, padding=1)
        nn.init.normal_(self.v_head.weight, 0.0, 0.005)
        nn.init.zeros_(self.v_head.bias)

    def forward(
        self,
        traces: torch.Tensor,
        source_x_m: torch.Tensor | None = None,
    ) -> WaveLatentOutput:
        """
        Args:
            traces: `[B, S, T=2001, R=70]` full-rate or `[B, S, T=251, R=70]` Nyquist-decimated recordings.
            source_x_m: Optional `[S]` source horizontal positions in meters (`0..690`).
        """
        if traces.ndim == 3:
            traces = traces.unsqueeze(0)
        traces_251 = traces[:, :, ::8, :] if traces.shape[2] > 251 else traces

        bsz, shots, nt, nrec = traces_251.shape
        device = traces_251.device
        dtype = traces_251.dtype

        if source_x_m is None:
            source_x_m = (
                torch.linspace(0.0, 690.0, shots, device=device, dtype=dtype)
                if shots > 1
                else torch.tensor([345.0], device=device, dtype=dtype)
            )

        # 3-channel physical wave stem: [True-Amp, TGC 15x, Time-Derivative du/dt]
        ch0 = torch.asinh(traces_251 / self.rms_scale)
        ch1 = torch.asinh((traces_251 * self.tgc_gain) / (0.12 * self.rms_scale))
        dt_diff = F.pad(traces_251[:, :, 1:, :] - traces_251[:, :, :-1, :], (0, 0, 1, 0))
        ch2 = torch.asinh(dt_diff / (0.25 * self.rms_scale))

        x = torch.stack([ch0, ch1, ch2], dim=2).reshape(bsz * shots, 3, nt, nrec)
        feat = self.stem(x)  # [B*S, 64, T'=63, R'=35]
        _, ch, nt_tok, nr_tok = feat.shape

        # Branch A: High-resolution learned column time-to-depth projection (T'=63 -> Z'=35)
        col_depth = (
            self.time_proj(feat.permute(0, 1, 3, 2))
            .permute(0, 1, 3, 2)
            .view(bsz, shots, ch, 35, 35)
        )

        # Branch B: Multi-Velocity NMO Hyperbola Bank (M=4 velocities across 1800..4050 m/s)
        rx = (torch.arange(35, device=device, dtype=dtype) + 0.5) * 20.0
        zm = (torch.arange(35, device=device, dtype=dtype) + 0.5) * 20.0
        half_off = 0.5 * (rx.view(1, 35, 1) - source_x_m.view(shots, 1, 1))  # [S, R'=35, 1]
        dist_2way = 2.0 * torch.sqrt(half_off.square() + zm.view(1, 1, 35).square())  # [S, R'=35, Z'=35]

        v_bank = torch.clamp(self.v_bank_mps, 1400.0, 4500.0)  # [M=4, Z'=35]
        tau_bank = dist_2way.unsqueeze(0) / v_bank.view(self.n_vel_bank, 1, 1, 35)  # [M, S, R', Z']
        t_grid = (torch.arange(nt_tok, device=device, dtype=dtype) + 0.5) / nt_tok
        w_nmo = torch.softmax(
            -220.0 * (tau_bank.unsqueeze(-2) - t_grid.view(1, 1, 1, nt_tok, 1)).square(),
            dim=-2,
        )  # [M, S, R', T', Z']

        nmo_bank = torch.einsum(
            "bsctr,msrtz->bsmczr",
            feat.view(bsz, shots, ch, nt_tok, nr_tok),
            w_nmo,
        ).reshape(bsz * shots, self.n_vel_bank * ch, 35, 35)
        nmo_depth = self.nmo_bank_proj(F.silu(nmo_bank)).view(bsz, shots, ch, 35, 35)

        depth_raw = (col_depth + self.nmo_scale * nmo_depth).reshape(bsz * shots, ch, 35, 35)
        off_map = ((rx.view(1, 1, 1, 35) - source_x_m.view(1, shots, 1, 1)) / 690.0).expand(
            bsz, shots, 35, 35
        ).reshape(bsz * shots, 1, 35, 35)
        src_map = ((source_x_m / 345.0 - 1.0).view(1, shots, 1, 1)).expand(
            bsz, shots, 35, 35
        ).reshape(bsz * shots, 1, 35, 35)
        depth_s = self.shot_refine(torch.cat([depth_raw, off_map, src_map], dim=1)).view(
            bsz, shots, ch, 35, 35
        )

        # Project arbitrary S shots onto K=5 virtual source modes
        dist_sk = (source_x_m.view(shots, 1) - self.virtual_xs.view(1, self.n_modes)) / 120.0
        w_sk = torch.exp(-0.5 * dist_sk.square())  # [S, K]
        illum_k = w_sk.sum(dim=0)                  # [K]
        w_sk_norm = w_sk / (illum_k.unsqueeze(0) + 1e-4)

        virtual_maps = torch.einsum("bsczx,sk->bkczx", depth_s, w_sk_norm).reshape(
            bsz, self.n_modes * ch, 35, 35
        )
        illum_map = illum_k.view(1, self.n_modes, 1, 1).expand(bsz, self.n_modes, 35, 35)
        coords_b = self.coords.expand(bsz, -1, -1, -1)

        h35 = F.silu(self.fuse(torch.cat([virtual_maps, illum_map, coords_b], dim=1)))
        f35 = self.fine35(h35)
        c18 = self.down18(f35)
        u35 = self.join35(
            torch.cat(
                [f35, F.interpolate(c18, size=(35, 35), mode="bilinear", align_corners=False)],
                dim=1,
            )
        )
        detail70 = self.up70(
            F.interpolate(u35, size=(70, 70), mode="bilinear", align_corners=False)
        )

        v_aux_km = (3.0 + 2.0 * torch.tanh(self.v_head(detail70) + self.base_atanh)).squeeze(1)
        z_coarse = self.zc_proj(u35)
        z_fine = self.zf_proj(detail70)

        return WaveLatentOutput(
            z_coarse=z_coarse,
            z_fine=z_fine,
            v_aux_km=v_aux_km,
            temporal_attn=w_nmo,
            spatial_attn=w_sk_norm,
        )


# Backwards-compatible alias
TraveltimeAttentionWaveEncoder = HybridVirtualSourceWaveEncoder


class StagedWaveToGaussianModel(nn.Module):
    """Combined Wave Encoder + Geological Latent Bridge + Triangle Gaussian Decoder."""

    def __init__(
        self,
        wave_encoder: HybridVirtualSourceWaveEncoder,
        gaussian_decoder: TriangleGaussianDecoder,
        geo_encoder: GeologyLatentEncoder | None = None,
    ) -> None:
        super().__init__()
        self.wave_encoder = wave_encoder
        self.gaussian_decoder = gaussian_decoder
        self.geo_encoder = geo_encoder

    def freeze_decoder(self, freeze: bool = True) -> None:
        for p in self.gaussian_decoder.parameters():
            p.requires_grad_(not freeze)
        if self.geo_encoder is not None:
            for p in self.geo_encoder.parameters():
                p.requires_grad_(False)

    def forward(
        self,
        traces: torch.Tensor,
        source_x_m: torch.Tensor | None = None,
        density_override: torch.Tensor | None = None,
    ) -> tuple[DecoderOutput, WaveLatentOutput]:
        wave_lat = self.wave_encoder(traces, source_x_m=source_x_m)
        if self.geo_encoder is not None:
            zc_geo, zf_geo = self.geo_encoder(wave_lat.v_aux_km)
            zc = 0.25 * wave_lat.z_coarse + 0.75 * zc_geo
            zf = 0.25 * wave_lat.z_fine + 0.75 * zf_geo
        else:
            zc = wave_lat.z_coarse
            zf = wave_lat.z_fine
        dec_out = self.gaussian_decoder(zc, zf, density_override=density_override)
        return dec_out, wave_lat
