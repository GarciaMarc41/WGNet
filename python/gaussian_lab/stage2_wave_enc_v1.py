"""Frozen v1 wave encoder (best: 327.29 m/s val RMSE, `results/staged-native/v1_327mps/`).

Kept verbatim so the v1 checkpoint stays loadable and serves as the ablation baseline
for the v2 encoder in `stage2_wave_enc.py`.
"""

from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .stage2_wave_enc import WaveLatentOutput


class PureResBlockV1(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        nn.init.kaiming_normal_(self.c1.weight, nonlinearity="relu")
        nn.init.zeros_(self.c1.bias)
        nn.init.kaiming_normal_(self.c2.weight, nonlinearity="relu")
        nn.init.zeros_(self.c2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x + 0.5 * self.c2(F.silu(self.c1(x))))


class HybridVirtualSourceWaveEncoderV1(nn.Module):
    """2-channel TGC stem (T'=32) + single-velocity NMO gather + K=5 virtual sources."""

    def __init__(
        self,
        mean_km: np.ndarray | torch.Tensor,
        mean_bg_traces=None,
        rms_scale: float = 1.505,
        scatter_scale: float = 0.903,
        n_modes: int = 5,
    ) -> None:
        super().__init__()
        del mean_bg_traces, scatter_scale
        self.rms_scale = float(rms_scale)
        self.n_modes = n_modes

        t_norm = torch.linspace(0.0, 1.0, 251).view(1, 1, 251, 1)
        self.register_buffer("tgc_gain", ((t_norm + 0.05) / 0.12).pow(1.4))

        mean_t = torch.as_tensor(mean_km, dtype=torch.float32).view(1, 1, 70, 70)
        self.register_buffer("base_atanh", torch.atanh(torch.clamp((mean_t - 3.0) * 0.5, -0.999, 0.999)))
        mean_35_mps = F.interpolate(mean_t, size=(35, 35), mode="bilinear", align_corners=False).squeeze() * 1000.0
        self.register_buffer("v_z_mps", mean_35_mps.mean(dim=1))

        self.stem = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)),
            nn.SiLU(),
            nn.Conv2d(32, 48, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1)),
            nn.SiLU(),
            nn.Conv2d(48, 64, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.SiLU(),
        )
        self.time_proj = nn.Linear(32, 35)
        nn.init.normal_(self.time_proj.weight, 0.0, math.sqrt(1.0 / 32.0))
        nn.init.zeros_(self.time_proj.bias)
        self.nmo_scale = nn.Parameter(torch.tensor(0.35))
        self.register_buffer("virtual_xs", torch.linspace(0.0, 690.0, n_modes))
        coords = torch.stack(
            torch.meshgrid(torch.linspace(-1.0, 1.0, 35), torch.linspace(-1.0, 1.0, 35), indexing="ij"),
            dim=0,
        ).unsqueeze(0)
        self.register_buffer("coords", coords)

        self.fuse = nn.Conv2d(n_modes * 64 + n_modes + 2, 96, kernel_size=1)
        self.fine35 = nn.Sequential(PureResBlockV1(96), PureResBlockV1(96))
        self.down18 = nn.Sequential(
            nn.Conv2d(96, 128, 3, stride=2, padding=1), nn.SiLU(), PureResBlockV1(128), PureResBlockV1(128)
        )
        self.join35 = nn.Sequential(
            nn.Conv2d(96 + 128, 64, 3, padding=1), nn.SiLU(), PureResBlockV1(64), PureResBlockV1(64)
        )
        self.up70 = nn.Sequential(
            nn.Conv2d(64, 48, 3, padding=1), nn.SiLU(), PureResBlockV1(48), nn.Conv2d(48, 48, 3, padding=1), nn.SiLU()
        )
        self.zc_proj = nn.Sequential(nn.Conv2d(64, 64, 1), nn.GroupNorm(8, 64))
        self.zf_proj = nn.Sequential(nn.Conv2d(48, 48, 1), nn.GroupNorm(8, 48))
        self.v_head = nn.Conv2d(48, 1, kernel_size=3, padding=1)
        nn.init.normal_(self.v_head.weight, 0.0, 0.005)
        nn.init.zeros_(self.v_head.bias)

    def forward(self, traces: torch.Tensor, source_x_m: torch.Tensor | None = None) -> WaveLatentOutput:
        if traces.ndim == 3:
            traces = traces.unsqueeze(0)
        traces_251 = traces[:, :, ::8, :] if traces.shape[2] > 251 else traces
        bsz, shots, nt, nrec = traces_251.shape
        device, dtype = traces_251.device, traces_251.dtype
        if source_x_m is None:
            source_x_m = (
                torch.linspace(0.0, 690.0, shots, device=device, dtype=dtype)
                if shots > 1
                else torch.tensor([345.0], device=device, dtype=dtype)
            )

        ch0 = torch.asinh(traces_251 / self.rms_scale)
        ch1 = torch.asinh((traces_251 * self.tgc_gain) / (0.12 * self.rms_scale))
        feat = self.stem(torch.stack([ch0, ch1], dim=2).reshape(bsz * shots, 2, nt, nrec))
        _, ch, nt_tok, nr_tok = feat.shape

        col_depth = self.time_proj(feat.permute(0, 1, 3, 2)).permute(0, 1, 3, 2).view(bsz, shots, ch, 35, 35)

        rx = (torch.arange(35, device=device, dtype=dtype) + 0.5) * 20.0
        zm = (torch.arange(35, device=device, dtype=dtype) + 0.5) * 20.0
        half_off = 0.5 * (rx.view(1, 35, 1) - source_x_m.view(shots, 1, 1))
        tau = 2.0 * torch.sqrt(half_off.square() + zm.view(1, 1, 35).square()) / self.v_z_mps.view(1, 1, 35)
        t_grid = (torch.arange(nt_tok, device=device, dtype=dtype) + 0.5) / nt_tok
        w_nmo = torch.softmax(-160.0 * (tau.unsqueeze(-2) - t_grid.view(1, 1, nt_tok, 1)).square(), dim=-2)
        nmo_depth = torch.einsum("bsctr,srtz->bsczr", feat.view(bsz, shots, ch, nt_tok, nr_tok), w_nmo)
        depth_s = col_depth + self.nmo_scale * nmo_depth

        dist_sk = (source_x_m.view(shots, 1) - self.virtual_xs.view(1, self.n_modes)) / 120.0
        w_sk = torch.exp(-0.5 * dist_sk.square())
        illum_k = w_sk.sum(dim=0)
        w_sk_norm = w_sk / (illum_k.unsqueeze(0) + 1e-4)
        virtual_maps = torch.einsum("bsczx,sk->bkczx", depth_s, w_sk_norm).reshape(bsz, self.n_modes * ch, 35, 35)
        illum_map = illum_k.view(1, self.n_modes, 1, 1).expand(bsz, self.n_modes, 35, 35)

        h35 = F.silu(self.fuse(torch.cat([virtual_maps, illum_map, self.coords.expand(bsz, -1, -1, -1)], dim=1)))
        f35 = self.fine35(h35)
        c18 = self.down18(f35)
        u35 = self.join35(torch.cat([f35, F.interpolate(c18, size=(35, 35), mode="bilinear", align_corners=False)], dim=1))
        detail70 = self.up70(F.interpolate(u35, size=(70, 70), mode="bilinear", align_corners=False))

        v_aux_km = (3.0 + 2.0 * torch.tanh(self.v_head(detail70) + self.base_atanh)).squeeze(1)
        return WaveLatentOutput(
            z_coarse=self.zc_proj(u35),
            z_fine=self.zf_proj(detail70),
            v_aux_km=v_aux_km,
            temporal_attn=w_nmo,
            spatial_attn=w_sk_norm,
        )
