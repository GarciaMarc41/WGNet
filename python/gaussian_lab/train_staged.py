"""3-Stage Training Curriculum, Resumable Checkpointing, and Legacy Baseline Loader (`train_staged.py`).

Implements:
- `LegacyDirectWaveGaussianNetwork`: Exact PyTorch port of `src/direct/network.ts` capable of loading
  the saved 633,547-parameter checkpoint in `results/direct-gaussians/Gaussian-direct-results.json`
  for direct counterfactual wave-attribution probing.
- `StagedTrainer`:
    * Stage 1: Train Geological Autoencoder (`v -> Z_geo -> Delaunay Gaussians -> v_hat`)
    * Stage 2: Freeze Gaussian Decoder; train Traveltime Cross-Attention Wave Encoder (`d -> Z_hat_geo`)
      with latent distillation + rendered field/edge/density supervision
    * Stage 3: Joint fine-tuning with online `wgpu` discrete wave-adjoint physics loss
"""

from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wave_gauss_rs

from .data import CELLS, N, NT, SHOTS, OpenFWIDatasetManager
from .probes import WaveAttributionSuite
from .renderer import (
    DENSITY_LOSS_WEIGHT,
    EDGE_LOSS_WEIGHT,
    FIELD_SIDE,
    compute_metrics_mps,
    density_supervision_loss,
    field_and_edge_loss,
    render_gaussians,
)
from .stage1_decoder import GeologicalAutoencoder, GeologyLatentEncoder, TriangleGaussianDecoder
from .stage2_wave_enc import (
    HybridVirtualSourceWaveEncoder,
    StagedWaveToGaussianModel,
    TraveltimeAttentionWaveEncoder,
)
from .stage3_physics import WavePhysicsLoss
from .stage2_wave_enc_v1 import HybridVirtualSourceWaveEncoderV1


def build_wave_encoder(encoder: str, data_mgr: OpenFWIDatasetManager) -> nn.Module:
    """`v1`: frozen 327 m/s baseline encoder. `v2`: multi-velocity NMO + axial mixer encoder."""
    rms_scale, scatter_scale, bg_traces = data_mgr.get_normalization_stats()
    cls = {"v1": HybridVirtualSourceWaveEncoderV1, "v2": HybridVirtualSourceWaveEncoder}[encoder]
    return cls(
        mean_km=data_mgr.train_mean_km,
        mean_bg_traces=bg_traces,
        rms_scale=rms_scale,
        scatter_scale=scatter_scale,
    )


def get_preferred_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class LegacyDirectWaveGaussianNetwork(nn.Module):
    """Exact PyTorch port of `DirectWaveGaussianNetwork` (`src/direct/network.ts`) for probing the saved checkpoint."""

    def __init__(
        self,
        mean_km: np.ndarray,
        scale: float,
        shots: int = SHOTS,
        nt: int = NT,
        nrec: int = N,
    ) -> None:
        super().__init__()
        self.shots = shots
        self.nt = nt
        self.nrec = nrec
        self.scale = float(scale)
        self.mean_km = np.asarray(mean_km, dtype=np.float32).reshape(FIELD_SIDE * FIELD_SIDE)

        # Register the exact 36 parameter tensors (18 weight+bias pairs) matching `src/direct/network.ts`
        temporal_cells = math.ceil(math.ceil(math.ceil(nt / 16) / 2) / 2)  # 32
        specs = [
            (16, 1, 1, 32, (16, 1)),            # conv 0
            (5, 3, 32, 48, (2, 2)),             # conv 1
            (3, 3, 48, 64, (2, 1)),             # conv 2
            (3, 3, 64, 64, (1, 1)),             # conv 3
        ]
        self.conv_strides: list[tuple[int, int]] = []
        self.tf_weights = nn.ParameterList()

        def add_pair(shape: tuple[int, ...]) -> int:
            idx = len(self.tf_weights)
            self.tf_weights.append(nn.Parameter(torch.zeros(shape, dtype=torch.float32)))
            self.tf_weights.append(nn.Parameter(torch.zeros((shape[-1],), dtype=torch.float32)))
            return idx

        self.conv_indices: list[int] = []
        for h, w, ic, oc, st in specs:
            self.conv_indices.append(add_pair((h, w, ic, oc)))
            self.conv_strides.append(st)

        self.time_proj_idx = add_pair((temporal_cells, 35))

        post_specs = [
            (1, 1, shots * 64 + 2, 64, (1, 1)),  # conv 4
            (3, 3, 64, 64, (1, 1)),              # conv 5
            (3, 3, 64, 64, (1, 1)),              # conv 6
            (3, 3, 64, 96, (2, 2)),              # conv 7
            (3, 3, 96, 96, (1, 1)),              # conv 8
            (3, 3, 96, 96, (1, 1)),              # conv 9
            (3, 3, 96 + 64, 64, (1, 1)),         # conv 10
            (3, 3, 64, 64, (1, 1)),              # conv 11
            (3, 3, 64, 64, (1, 1)),              # conv 12
            (3, 3, 64, 48, (1, 1)),              # conv 13
            (3, 3, 48, 48, (1, 1)),              # conv 14
            (3, 3, 48, 1, (1, 1)),               # conv 15 (densityHead)
        ]
        for h, w, ic, oc, st in post_specs:
            self.conv_indices.append(add_pair((h, w, ic, oc)))
            self.conv_strides.append(st)

        self.attr_hidden_idx = add_pair((4 * 48 + 5, 64))
        self.attr_head_idx = add_pair((64, 7))

        coords = np.zeros((1, 35, 35, 2), dtype=np.float32)
        for z in range(35):
            for x in range(35):
                coords[0, z, x, 0] = x / 17.0 - 1.0
                coords[0, z, x, 1] = z / 17.0 - 1.0
        self.register_buffer("coordinates", torch.from_numpy(coords))

    def load_flat_weights(self, flat: np.ndarray) -> None:
        total = sum(p.numel() for p in self.tf_weights)
        assert len(flat) == total, f"Expected {total} weights, got {len(flat)}"
        offset = 0
        with torch.no_grad():
            for p in self.tf_weights:
                cnt = p.numel()
                chunk = torch.from_numpy(
                    np.asarray(flat[offset : offset + cnt], dtype=np.float32)
                ).view(p.shape)
                p.copy_(chunk)
                offset += cnt

    def _tf_same_conv2d(self, x_nhwc: torch.Tensor, conv_slot: int) -> torch.Tensor:
        w_idx = self.conv_indices[conv_slot]
        sh, sw = self.conv_strides[conv_slot]
        w_hwio = self.tf_weights[w_idx]
        bias = self.tf_weights[w_idx + 1]
        kh, kw, _, _ = w_hwio.shape
        _, in_h, in_w, _ = x_nhwc.shape

        out_h = math.ceil(in_h / sh)
        out_w = math.ceil(in_w / sw)
        pad_h = max((out_h - 1) * sh + kh - in_h, 0)
        pad_w = max((out_w - 1) * sw + kw - in_w, 0)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        x_nchw = x_nhwc.permute(0, 3, 1, 2)
        if pad_h > 0 or pad_w > 0:
            x_nchw = F.pad(x_nchw, (pad_left, pad_right, pad_top, pad_bottom))
        w_oihw = w_hwio.permute(3, 2, 0, 1).contiguous()
        y_nchw = F.conv2d(x_nchw, w_oihw, bias=bias, stride=(sh, sw))
        return y_nchw.permute(0, 2, 3, 1).contiguous()

    def _residual(self, x: torch.Tensor, first_slot: int) -> torch.Tensor:
        h = F.silu(self._tf_same_conv2d(x, first_slot))
        h = self._tf_same_conv2d(h, first_slot + 1)
        return F.silu(x + 0.5 * h)

    @staticmethod
    def _tf_resize_bilinear(x_nhwc: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """Exact reproduction of TensorFlow.js `tf.image.resizeBilinear(x, size, alignCorners=false)`
        which uses `halfPixelCenters=false`: source coordinate `y_in = y_out * (in_h / out_h)`.
        """
        _, in_h, in_w, _ = x_nhwc.shape
        out_h, out_w = size
        device = x_nhwc.device
        dtype = x_nhwc.dtype

        yf = torch.arange(out_h, device=device, dtype=dtype) * (in_h / out_h)
        xf = torch.arange(out_w, device=device, dtype=dtype) * (in_w / out_w)
        y0 = torch.floor(yf).long().clamp(0, in_h - 1)
        x0 = torch.floor(xf).long().clamp(0, in_w - 1)
        y1 = (y0 + 1).clamp(0, in_h - 1)
        x1 = (x0 + 1).clamp(0, in_w - 1)
        ly = (yf - y0.to(dtype)).view(1, out_h, 1, 1)
        lx = (xf - x0.to(dtype)).view(1, 1, out_w, 1)

        v00 = x_nhwc[:, y0][:, :, x0]
        v01 = x_nhwc[:, y0][:, :, x1]
        v10 = x_nhwc[:, y1][:, :, x0]
        v11 = x_nhwc[:, y1][:, :, x1]
        top = v00 + (v01 - v00) * lx
        bot = v10 + (v11 - v10) * lx
        return (top + (bot - top) * ly).contiguous()

    def forward_maps(self, traces_1example: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run wave encoder maps for a single `[5, 2001, 70]` recording -> `(detail [4900, 48], density [70, 70])`."""
        inp = torch.asinh(traces_1example / self.scale).view(self.shots, self.nt, self.nrec, 1)
        feat = F.silu(self._tf_same_conv2d(inp, 0))
        feat = F.silu(self._tf_same_conv2d(feat, 1))
        feat = F.silu(self._tf_same_conv2d(feat, 2))
        feat = F.silu(self._tf_same_conv2d(feat, 3))
        t_cells = feat.shape[1]

        w_tp = self.tf_weights[self.time_proj_idx]
        b_tp = self.tf_weights[self.time_proj_idx + 1]
        depth = (
            torch.matmul(feat.permute(0, 2, 3, 1).reshape(self.shots * 35 * 64, t_cells), w_tp)
            + b_tp
        )
        depth = (
            depth.view(self.shots, 35, 64, 35)
            .permute(3, 1, 0, 2)
            .reshape(1, 35, 35, self.shots * 64)
        )
        fused = F.silu(self._tf_same_conv2d(torch.cat([depth, self.coordinates], dim=3), 4))
        fine = self._residual(fused, 5)
        coarse = self._residual(F.silu(self._tf_same_conv2d(fine, 7)), 8)
        up = self._tf_resize_bilinear(coarse, (35, 35))
        joined = F.silu(self._tf_same_conv2d(torch.cat([fine, up], dim=3), 10))
        decoded = self._residual(joined, 11)
        full = self._tf_resize_bilinear(decoded, (70, 70))
        detail = F.silu(self._tf_same_conv2d(F.silu(self._tf_same_conv2d(full, 13)), 14))
        logits = self._tf_same_conv2d(detail, 15).view(70, 70)
        return detail.view(4900, 48), torch.sigmoid(logits)

    def predict_batch(self, traces_batch: torch.Tensor) -> torch.Tensor:
        """Predict rendered `[B, 70, 70]` velocity fields in km/s for a batch of `[B, 5, 2001, 70]` recordings."""
        if traces_batch.ndim == 3:
            traces_batch = traces_batch.unsqueeze(0)
        outs = []
        device = traces_batch.device
        for b in range(traces_batch.shape[0]):
            detail_flat, density = self.forward_maps(traces_batch[b])
            ppm_np = np.ascontiguousarray(density.detach().cpu().numpy().reshape(-1), dtype=np.float32)
            verts_np, priors_np, sample_np, _, count = wave_gauss_rs.density_to_layout(ppm_np, 70, 70)

            # Exact bilinear lookup matching `src/direct/network.ts`
            indices = np.zeros((count, 16), dtype=np.int64)
            coeffs = np.zeros((count, 16, 1), dtype=np.float32)
            base = np.zeros((count, 1), dtype=np.float32)

            def bilinear_np(x: float, z: float):
                fx = max(0.0, min(69.0, x - 0.5))
                fz = max(0.0, min(69.0, z - 0.5))
                x0 = int(math.floor(fx))
                z0 = int(math.floor(fz))
                x1 = min(69, x0 + 1)
                z1 = min(69, z0 + 1)
                dx = fx - x0
                dz = fz - z0
                return (
                    [z0 * 70 + x0, z0 * 70 + x1, z1 * 70 + x0, z1 * 70 + x1],
                    [(1 - dx) * (1 - dz), dx * (1 - dz), (1 - dx) * dz, dx * dz],
                )

            sc_flat = sample_np.reshape(count, 8)
            for k in range(count):
                for pt in range(4):
                    ids, wts = bilinear_np(float(sc_flat[k, pt * 2]), float(sc_flat[k, pt * 2 + 1]))
                    indices[k, pt * 4 : (pt + 1) * 4] = ids
                    coeffs[k, pt * 4 : (pt + 1) * 4, 0] = wts
                ids, wts = bilinear_np(float(priors_np[k, 0]), float(priors_np[k, 1]))
                m_val = sum(self.mean_km[idx] * wts[j] for j, idx in enumerate(ids))
                base[k, 0] = math.atanh(max(-0.999, min(0.999, (m_val - 3.0) / 2.0)))

            idx_t = torch.from_numpy(indices.reshape(-1)).to(device=device)
            coeff_t = torch.from_numpy(coeffs).to(device=device)
            verts_t = torch.from_numpy(verts_np).to(device=device)
            priors_t = torch.from_numpy(priors_np).to(device=device)
            base_t = torch.from_numpy(base).to(device=device)

            gathered = detail_flat[idx_t].view(count, 16, 48)
            samples = (gathered * coeff_t).view(count, 4, 4, 48).sum(dim=2).reshape(count, 192)
            geom_feat = priors_t / torch.tensor([35.0, 35.0, math.pi, 4.0, 4.0], device=device) - torch.tensor(
                [1.0, 1.0, 0.0, 0.0, 0.0], device=device
            )
            mlp_in = torch.cat([samples, geom_feat], dim=1)
            w_h = self.tf_weights[self.attr_hidden_idx]
            b_h = self.tf_weights[self.attr_hidden_idx + 1]
            hidden = F.silu(torch.matmul(mlp_in, w_h) + b_h)
            w_o = self.tf_weights[self.attr_head_idx]
            b_o = self.tf_weights[self.attr_head_idx + 1]
            raw = torch.matmul(hidden, w_o) + b_o

            bary = F.softmax(raw[:, 0:3], dim=1).unsqueeze(-1)
            pos = (bary * verts_t).sum(dim=1)
            widths = torch.clamp(
                priors_t[:, 3:5] + torch.tanh(raw[:, 3:5]) * 1.5,
                math.log(0.05),
                math.log(140.0),
            )
            angle = priors_t[:, 2:3] + torch.tanh(raw[:, 5:6]) * (math.pi * 0.5)
            val = torch.tanh(raw[:, 6:7] + base_t) * 2.0 + 3.0
            params = torch.cat([pos, angle, widths, val], dim=1)
            outs.append(render_gaussians(params, height=70, width=70))
        return torch.stack(outs, dim=0)


class StagedTrainer:
    """Manages Stage 1, Stage 2, and Stage 3 training with live-optimizer resumable checkpoints."""

    def __init__(
        self,
        data_mgr: OpenFWIDatasetManager,
        checkpoint_dir: Path | str | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.data_mgr = data_mgr
        self.device = device or get_preferred_device()
        self.ckpt_dir = (
            Path(checkpoint_dir)
            if checkpoint_dir is not None
            else (self.data_mgr.root_dir / "results" / "staged-native")
        )
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def train_stage1_geology_decoder(
        self,
        steps: int = 2000,
        batch_size: int = 8,
        lr: float = 1e-3,
        val_every: int = 200,
        seed: int = 7201,
        resume: bool = True,
    ) -> GeologicalAutoencoder:
        """Train Stage 1 Geological Autoencoder (`v -> Z_geo -> Delaunay Gaussians -> v_hat`)."""
        model = GeologicalAutoencoder(mean_km=self.data_mgr.train_mean_km).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, steps), eta_min=lr * 0.1
        )

        ckpt_path = self.ckpt_dir / "stage1_checkpoint.pt"
        start_step = 0
        best_val_rmse = float("inf")
        best_state = None
        history: list[dict] = []

        if resume and ckpt_path.exists():
            saved = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            model.load_state_dict(saved["live_model"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            start_step = int(saved["step"])
            best_val_rmse = float(saved["best_val_rmse"])
            best_state = saved["best_model"]
            history = saved.get("history", [])
            print(
                f"[Stage 1] Resumed checkpoint at step {start_step}/{steps} "
                f"(best_val_RMSE={best_val_rmse:.2f} m/s)",
                flush=True,
            )

        train_fields = self.data_mgr.fields_km["train"]
        teacher_densities = self.data_mgr.get_train_teacher_densities()
        val_ids = self.data_mgr.get_validation_cohort_indices(per_family=2)
        val_fields = torch.from_numpy(self.data_mgr.fields_km["validation"][val_ids]).to(self.device)

        rng = np.random.default_rng(seed + start_step)
        n_train = len(train_fields)
        t0 = time.perf_counter()

        for step in range(start_step + 1, steps + 1):
            model.train()
            batch_idx = rng.choice(n_train, size=batch_size, replace=False)
            v_batch = torch.from_numpy(train_fields[batch_idx]).to(self.device)
            t_dens = torch.from_numpy(teacher_densities[batch_idx]).to(self.device)

            # Warm-up first 20% of steps with teacher density for stable triangle layout, then use predicted PPM
            use_teacher_layout = step <= max(50, int(0.15 * steps))
            out = model(v_batch, density_override=t_dens if use_teacher_layout else None)

            geo_loss, field_mse, edge_mse = field_and_edge_loss(out.field_km, v_batch)
            dens_loss = density_supervision_loss(out.logits, out.density, t_dens)
            loss = geo_loss + DENSITY_LOSS_WEIGHT * dens_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

            if step % val_every == 0 or step == steps or step == 1:
                model.eval()
                with torch.no_grad():
                    val_out = model(val_fields)
                    val_metrics = compute_metrics_mps(val_out.field_km, val_fields)
                    val_rmse = val_metrics["field_rmse_mps"]
                    val_grad = val_metrics["grad_rmse_mps"]
                    mean_cnt = float(val_out.counts.float().mean().item())

                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

                rec = {
                    "step": step,
                    "train_loss": float(loss.item()),
                    "val_field_rmse_mps": val_rmse,
                    "val_grad_rmse_mps": val_grad,
                    "val_mean_gaussians": mean_cnt,
                    "elapsed_sec": time.perf_counter() - t0,
                }
                history.append(rec)
                print(
                    f"[Stage 1 | Step {step}/{steps}] "
                    f"val_RMSE={val_rmse:.2f} m/s (best={best_val_rmse:.2f}) | "
                    f"grad_RMSE={val_grad:.2f} m/s | Gaussians={mean_cnt:.1f}"
                )
                torch.save(
                    {
                        "step": step,
                        "complete": step == steps,
                        "best_val_rmse": best_val_rmse,
                        "best_model": best_state,
                        "live_model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "history": history,
                    },
                    ckpt_path,
                )

        if best_state is not None:
            model.load_state_dict(best_state)
        return model

    def train_stage2_wave_controller(
        self,
        stage1_model: GeologicalAutoencoder,
        steps: int = 2400,
        batch_size: int = 16,
        lr: float = 8e-4,
        latent_weight: float = 0.20,
        val_every: int = 200,
        seed: int = 7202,
        resume: bool = True,
        encoder: str = "v2",
    ) -> StagedWaveToGaussianModel:
        """Train Stage 2 HybridVirtualSourceWaveEncoder (`d_waves -> (v_aux, Z_hat_geo)`) controlling Stage 1 Decoder."""
        stage1_model.eval()
        for p in stage1_model.parameters():
            p.requires_grad_(False)

        memmaps = self.data_mgr.ensure_wave_cache(splits_to_simulate=("train", "validation"))
        rms_scale, _, _ = self.data_mgr.get_normalization_stats()
        wave_enc = build_wave_encoder(encoder, self.data_mgr).to(self.device)

        # Clone encoder & decoder from Stage 1 and keep them frozen during Stage 2
        geo_enc = GeologyLatentEncoder().to(self.device)
        geo_enc.load_state_dict(stage1_model.encoder.state_dict())
        decoder = TriangleGaussianDecoder(mean_km=self.data_mgr.train_mean_km).to(self.device)
        decoder.load_state_dict(stage1_model.decoder.state_dict())
        staged_model = StagedWaveToGaussianModel(wave_enc, decoder, geo_encoder=geo_enc).to(
            self.device
        )
        staged_model.freeze_decoder(True)

        optimizer = torch.optim.AdamW(
            staged_model.wave_encoder.parameters(), lr=lr, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, steps), eta_min=5e-5
        )

        ckpt_path = self.ckpt_dir / f"stage2_{encoder}_checkpoint.pt"
        start_step = 0
        best_val_rmse = float("inf")
        best_state = None
        history: list[dict] = []

        if resume and ckpt_path.exists():
            try:
                saved = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                staged_model.load_state_dict(saved["live_model"])
                staged_model.freeze_decoder(True)
                optimizer.load_state_dict(saved["optimizer"])
                start_step = int(saved["step"])
                best_val_rmse = float(saved["best_val_rmse"])
                best_state = saved["best_model"]
                history = saved.get("history", [])
                if start_step < steps:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = 2.0e-4
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=max(1, steps - start_step), eta_min=2e-5
                    )
                else:
                    scheduler.load_state_dict(saved["scheduler"])
                print(
                    f"[Stage 2] Resumed checkpoint at step {start_step}/{steps} "
                    f"(best_val_RMSE={best_val_rmse:.2f} m/s)",
                    flush=True,
                )
            except Exception:
                print(
                    f"[Stage 2] {ckpt_path.name} does not match the '{encoder}' encoder; starting Stage 2 fresh.",
                    flush=True,
                )
                start_step = 0

        # Load Nyquist-decimated (T=251, dt=4ms) training waves into GPU memory (351 MB) for 6x faster training
        train_waves_gpu = torch.from_numpy(
            np.array(memmaps["train"][:, :, ::8, :], dtype=np.float32)
        ).to(self.device)
        train_fields_gpu = torch.from_numpy(self.data_mgr.fields_km["train"]).to(self.device)

        val_ids = self.data_mgr.get_validation_cohort_indices(per_family=2)
        val_fields = torch.from_numpy(self.data_mgr.fields_km["validation"][val_ids]).to(self.device)
        val_waves_251 = torch.from_numpy(
            np.array(memmaps["validation"][val_ids, :, ::8, :], dtype=np.float32)
        ).to(self.device)

        all_shot_x = torch.linspace(0.0, 690.0, 5, device=self.device)
        rng = np.random.default_rng(seed + start_step)
        n_train = len(train_fields_gpu)
        t0 = time.perf_counter()

        for step in range(start_step + 1, steps + 1):
            staged_model.wave_encoder.train()
            batch_idx = torch.from_numpy(rng.choice(n_train, size=batch_size, replace=False)).to(
                self.device
            )
            v_batch = train_fields_gpu.index_select(0, batch_idx)
            w_batch = train_waves_gpu.index_select(0, batch_idx)

            # Add 0-1.5% RMS waveform augmentation noise during training
            if rng.random() < 0.4:
                noise_std = float(rng.uniform(0.0, 0.015)) * rms_scale
                w_batch = w_batch + torch.randn_like(w_batch) * noise_std

            # Shot-subset curriculum (20% of steps train on 1..4 random shots for acquisition-agnostic generalization)
            if step > 400 and rng.random() < 0.20:
                n_keep = int(rng.integers(1, 5))
                keep_ids = np.sort(rng.choice(5, size=n_keep, replace=False))
                keep_t = torch.from_numpy(keep_ids).to(self.device)
                w_in = w_batch.index_select(1, keep_t)
                sx_in = all_shot_x.index_select(0, keep_t)
            else:
                w_in = w_batch
                sx_in = all_shot_x

            with torch.no_grad():
                target_z_coarse, target_z_fine = stage1_model.encoder(v_batch)

            wave_lat = staged_model.wave_encoder(w_in, source_x_m=sx_in)
            err = wave_lat.v_aux_km - v_batch
            dx = err[:, :, 1:] - err[:, :, :-1]
            dz = err[:, 1:, :] - err[:, :-1, :]
            geo_loss = (
                err.square().mean()
                + 0.15 * F.smooth_l1_loss(wave_lat.v_aux_km, v_batch, beta=0.05)
                + 0.35 * 0.5 * (dx.square().mean() + dz.square().mean())
                + 0.15 * 0.5 * (dx.abs().mean() + dz.abs().mean())
            )
            latent_loss = (
                F.mse_loss(wave_lat.z_coarse, target_z_coarse)
                + F.mse_loss(wave_lat.z_fine, target_z_fine)
            )
            loss = geo_loss + latent_weight * latent_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(staged_model.wave_encoder.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

            if step % val_every == 0 or step == steps or step == 1:
                staged_model.eval()
                with torch.no_grad():
                    val_dec, val_lat = staged_model(val_waves_251, source_x_m=all_shot_x)
                    val_metrics = compute_metrics_mps(val_dec.field_km, val_fields)
                    aux_metrics = compute_metrics_mps(val_lat.v_aux_km, val_fields)
                    val_rmse = val_metrics["field_rmse_mps"]
                    val_grad = val_metrics["grad_rmse_mps"]
                    aux_rmse = aux_metrics["field_rmse_mps"]
                    mean_cnt = float(val_dec.counts.float().mean().item())

                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    best_state = {
                        k: v.detach().cpu().clone() for k, v in staged_model.state_dict().items()
                    }

                rec = {
                    "step": step,
                    "train_loss": float(loss.item()),
                    "latent_loss": float(latent_loss.item()),
                    "val_field_rmse_mps": val_rmse,
                    "val_aux_rmse_mps": aux_rmse,
                    "val_grad_rmse_mps": val_grad,
                    "val_mean_gaussians": mean_cnt,
                    "elapsed_sec": time.perf_counter() - t0,
                }
                history.append(rec)
                print(
                    f"[Stage 2 | Step {step}/{steps}] "
                    f"Gauss_RMSE={val_rmse:.2f} m/s (best={best_val_rmse:.2f}) | "
                    f"Aux_RMSE={aux_rmse:.2f} m/s | grad_RMSE={val_grad:.2f} m/s | "
                    f"Gaussians={mean_cnt:.1f}",
                    flush=True,
                )
                torch.save(
                    {
                        "step": step,
                        "complete": step == steps,
                        "best_val_rmse": best_val_rmse,
                        "best_model": best_state,
                        "live_model": staged_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "history": history,
                    },
                    ckpt_path,
                )

        if best_state is not None:
            staged_model.load_state_dict(best_state)
        return staged_model

    def train_stage3_physics_guided(
        self,
        staged_model: StagedWaveToGaussianModel,
        steps: int = 400,
        batch_size: int = 4,
        lr: float = 2.5e-5,
        wave_loss_weight: float = 0.15,
        physics_every: int = 2,
        val_every: int = 50,
        seed: int = 7203,
        encoder: str = "v2",
    ) -> StagedWaveToGaussianModel:
        """Stage 3: Unfreeze Gaussian Decoder and fine-tune jointly with online `wgpu` wave-equation adjoint loss."""
        staged_model.freeze_decoder(False)
        phys_loss_fn = WavePhysicsLoss(use_stochastic_single_shot=True)

        optimizer = torch.optim.AdamW(
            [
                {"params": staged_model.wave_encoder.parameters(), "lr": lr},
                {"params": staged_model.gaussian_decoder.parameters(), "lr": lr * 0.25},
            ],
            weight_decay=1e-5,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, steps), eta_min=lr * 0.1
        )

        ckpt_path = self.ckpt_dir / f"stage3_{encoder}_checkpoint.pt"
        memmaps = self.data_mgr.ensure_wave_cache(splits_to_simulate=("train", "validation"))
        train_waves = memmaps["train"]
        train_fields = self.data_mgr.fields_km["train"]
        teacher_densities = self.data_mgr.get_train_teacher_densities()

        val_ids = self.data_mgr.get_validation_cohort_indices(per_family=2)
        val_fields = torch.from_numpy(self.data_mgr.fields_km["validation"][val_ids]).to(self.device)
        val_waves = torch.from_numpy(np.array(memmaps["validation"][val_ids], dtype=np.float32)).to(
            self.device
        )

        with torch.no_grad():
            staged_model.eval()
            init_dec, _ = staged_model(val_waves)
            best_val_rmse = compute_metrics_mps(init_dec.field_km, val_fields)["field_rmse_mps"]
            best_state = {k: v.detach().cpu().clone() for k, v in staged_model.state_dict().items()}

        rng = np.random.default_rng(seed)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        n_train = len(train_fields)
        history: list[dict] = []
        t0 = time.perf_counter()

        for step in range(1, steps + 1):
            staged_model.train()
            batch_idx = rng.choice(n_train, size=batch_size, replace=False)
            v_batch = torch.from_numpy(train_fields[batch_idx]).to(self.device)
            w_batch = torch.from_numpy(np.array(train_waves[batch_idx], dtype=np.float32)).to(
                self.device
            )
            t_dens = torch.from_numpy(teacher_densities[batch_idx]).to(self.device)

            dec_out, wave_lat = staged_model(w_batch)
            geo_loss, _, _ = field_and_edge_loss(dec_out.field_km, v_batch)
            aux_loss, _, _ = field_and_edge_loss(wave_lat.v_aux_km, v_batch)
            dens_loss = density_supervision_loss(dec_out.logits, dec_out.density, t_dens)
            base_loss = 0.5 * geo_loss + 0.5 * aux_loss + DENSITY_LOSS_WEIGHT * dens_loss

            if step % physics_every == 0:
                w_loss = phys_loss_fn(dec_out.field_km, w_batch, full_survey=False, generator=gen)
                loss = base_loss + wave_loss_weight * w_loss
                w_loss_val = float(w_loss.item())
            else:
                loss = base_loss
                w_loss_val = 0.0

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(staged_model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

            if step % val_every == 0 or step == steps:
                staged_model.eval()
                with torch.no_grad():
                    val_dec, _ = staged_model(val_waves)
                    val_metrics = compute_metrics_mps(val_dec.field_km, val_fields)
                    val_rmse = val_metrics["field_rmse_mps"]
                    val_grad = val_metrics["grad_rmse_mps"]

                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    best_state = {
                        k: v.detach().cpu().clone() for k, v in staged_model.state_dict().items()
                    }

                rec = {
                    "step": step,
                    "train_loss": float(loss.item()),
                    "wave_phys_loss": w_loss_val,
                    "val_field_rmse_mps": val_rmse,
                    "val_grad_rmse_mps": val_grad,
                    "elapsed_sec": time.perf_counter() - t0,
                }
                history.append(rec)
                print(
                    f"[Stage 3 | Step {step}/{steps}] "
                    f"val_RMSE={val_rmse:.2f} m/s (best={best_val_rmse:.2f}) | "
                    f"grad_RMSE={val_grad:.2f} m/s | wave_loss={w_loss_val:.5f}"
                )
                torch.save(
                    {
                        "step": step,
                        "complete": step == steps,
                        "best_val_rmse": best_val_rmse,
                        "best_model": best_state,
                        "live_model": staged_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "history": history,
                    },
                    ckpt_path,
                )

        if best_state is not None:
            staged_model.load_state_dict(best_state)
        return staged_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the 3-stage Rust + PyTorch Wave-to-Gaussian architecture."
    )
    parser.add_argument(
        "--stage",
        choices=["1", "2", "3", "all"],
        default="all",
        help="Which training stage to run (1: Geology Decoder, 2: Attention Wave Controller, 3: Physics Fine-Tuning, all: 1->2->3).",
    )
    parser.add_argument("--stage1-steps", type=int, default=2000)
    parser.add_argument("--stage2-steps", type=int, default=2400)
    parser.add_argument("--stage3-steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-every", type=int, default=200)
    parser.add_argument("--no-resume", action="store_true", help="Start fresh instead of resuming saved checkpoints.")
    parser.add_argument(
        "--encoder",
        choices=["v1", "v2"],
        default="v2",
        help="Wave encoder: v1 (327 m/s baseline) or v2 (multi-velocity NMO + axial mixer). Each has its own checkpoint.",
    )
    args = parser.parse_args()

    data_mgr = OpenFWIDatasetManager()
    trainer = StagedTrainer(data_mgr=data_mgr)
    resume = not args.no_resume

    stage1_ckpt = trainer.ckpt_dir / "stage1_checkpoint.pt"
    stage2_ckpt = trainer.ckpt_dir / f"stage2_{args.encoder}_checkpoint.pt"

    if args.stage in ("1", "all"):
        stage1_model = trainer.train_stage1_geology_decoder(
            steps=args.stage1_steps,
            batch_size=min(8, args.batch_size),
            val_every=args.val_every,
            resume=resume,
        )
    else:
        stage1_model = GeologicalAutoencoder(mean_km=data_mgr.train_mean_km).to(trainer.device)
        saved1 = torch.load(stage1_ckpt, map_location=trainer.device, weights_only=False)
        stage1_model.load_state_dict(saved1["best_model"])

    if args.stage in ("2", "all"):
        staged_model = trainer.train_stage2_wave_controller(
            stage1_model=stage1_model,
            steps=args.stage2_steps,
            batch_size=args.batch_size,
            val_every=args.val_every,
            resume=resume,
            encoder=args.encoder,
        )
    elif args.stage == "3":
        wave_enc = build_wave_encoder(args.encoder, data_mgr).to(trainer.device)
        geo_enc = GeologyLatentEncoder().to(trainer.device)
        decoder = TriangleGaussianDecoder(mean_km=data_mgr.train_mean_km).to(trainer.device)
        staged_model = StagedWaveToGaussianModel(
            wave_enc, decoder, geo_encoder=geo_enc
        ).to(trainer.device)
        saved2 = torch.load(stage2_ckpt, map_location=trainer.device, weights_only=False)
        staged_model.load_state_dict(saved2["best_model"])

    if args.stage in ("3", "all"):
        trainer.train_stage3_physics_guided(
            staged_model=staged_model,
            steps=args.stage3_steps,
            batch_size=max(2, min(4, args.batch_size // 4)),
            val_every=max(20, args.val_every // 4),
            encoder=args.encoder,
        )


if __name__ == "__main__":
    main()

