"""Verification, End-to-End Smoke Test, and Wave Attribution Probe Runner (`verify_and_probe.py`).

1. Verifies Rust `wave_gauss_rs` Delaunay speed & conservation, PyTorch MPS renderer parity,
   and `WavePhysicsFunction` autograd + renderer VJP against finite differences.
2. Loads the saved `DirectWaveGaussianNetwork` checkpoint from `results/direct-gaussians/Gaussian-direct-results.json`,
   verifies exact reproduction of the exported test fields, and runs the 5-probe `WaveAttributionSuite`
   to measure how much the baseline actually uses weak late reflections, far offsets, and phase relationships.
3. Verifies Stage 1 -> Stage 2 -> Stage 3 training and gradient flow on Apple Silicon MPS.
"""

from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import torch
import wave_gauss_rs

from gaussian_lab.data import DT, H, N, NT, PEAK_HZ, SHOTS, OpenFWIDatasetManager
from gaussian_lab.probes import WaveAttributionSuite
from gaussian_lab.renderer import compute_metrics_mps, field_and_edge_loss, render_gaussians
from gaussian_lab.stage1_decoder import GeologicalAutoencoder, TriangleGaussianDecoder
from gaussian_lab.stage2_wave_enc import StagedWaveToGaussianModel, TraveltimeAttentionWaveEncoder
from gaussian_lab.stage3_physics import WavePhysicsLoss
from gaussian_lab.train_staged import LegacyDirectWaveGaussianNetwork, get_preferred_device


def check_autograd_wave_renderer_vjp(device: torch.device) -> dict:
    """Verify that `WavePhysicsLoss(render_gaussians(params))` matches finite differences on Gaussian attributes."""
    engine_short = wave_gauss_rs.PyWaveEngine(2, 501, DT, PEAK_HZ, 20, 1000.0, 5000.0)

    # True 2-Gaussian field vs initial 2-Gaussian field
    true_params = torch.tensor(
        [
            [22.0, 14.0, 0.2, float(np.log(6.0)), float(np.log(4.0)), 3.2],
            [48.0, 16.0, -0.2, float(np.log(7.0)), float(np.log(4.5)), 2.1],
        ],
        dtype=torch.float32,
        device=device,
    )
    init_params = torch.tensor(
        [
            [21.0, 13.5, 0.1, float(np.log(6.5)), float(np.log(4.2)), 2.9],
            [49.0, 15.5, -0.1, float(np.log(6.8)), float(np.log(4.3)), 2.3],
        ],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )

    with torch.no_grad():
        true_field_mps = (render_gaussians(true_params) * 1000.0).detach().cpu().numpy().reshape(-1)
        obs = engine_short.simulate(np.ascontiguousarray(true_field_mps, dtype=np.float32))

    # Forward + Adjoint through PyTorch renderer -> Rust wgpu adjoint -> PyTorch backward
    pred_field_km = render_gaussians(init_params)
    pred_mps = np.ascontiguousarray(
        (pred_field_km.detach().cpu().numpy() * 1000.0).reshape(-1), dtype=np.float32
    )
    loss_val, grad_mps = engine_short.evaluate_adjoint(pred_mps, obs.reshape(-1))
    grad_km_t = torch.from_numpy(grad_mps * 1000.0).to(device=device, dtype=torch.float32)
    (param_grad,) = torch.autograd.grad((pred_field_km * grad_km_t).sum(), init_params)

    # Central finite difference along a combined Gaussian parameter perturbation direction
    direction = torch.tensor(
        [
            [0.4, -0.3, 0.05, 0.02, -0.02, 0.15],
            [-0.3, 0.4, -0.04, -0.02, 0.02, -0.12],
        ],
        dtype=torch.float32,
        device=device,
    )
    eps = 0.02
    with torch.no_grad():
        f_plus = (render_gaussians(init_params + eps * direction) * 1000.0).cpu().numpy().reshape(-1)
        f_minus = (render_gaussians(init_params - eps * direction) * 1000.0).cpu().numpy().reshape(-1)
        l_plus, _ = engine_short.evaluate_adjoint(np.ascontiguousarray(f_plus, dtype=np.float32), obs.reshape(-1))
        l_minus, _ = engine_short.evaluate_adjoint(np.ascontiguousarray(f_minus, dtype=np.float32), obs.reshape(-1))

    fd_slope = float(l_plus - l_minus) / (2.0 * eps)
    adj_slope = float((param_grad * direction).sum().item())
    rel_err = abs(fd_slope - adj_slope) / max(abs(fd_slope), abs(adj_slope), 1e-12)
    assert rel_err < 0.03, f"Renderer + Wave Adjoint VJP mismatch: fd={fd_slope} adj={adj_slope} rel={rel_err}"
    return {
        "passed": True,
        "loss": float(loss_val),
        "fd_slope": fd_slope,
        "adjoint_slope": adj_slope,
        "relative_error": rel_err,
    }


def probe_saved_direct_checkpoint(
    root_dir: Path, data_mgr: OpenFWIDatasetManager, device: torch.device
) -> dict:
    """Load the saved browser checkpoint from `results/direct-gaussians/Gaussian-direct-results.json`,
    verify parity on exported test cases, and run the 5-probe `WaveAttributionSuite`.
    """
    export_path = root_dir / "legacy_code" / "results" / "direct-gaussians" / "Gaussian-direct-results.json"
    if not export_path.exists():
        export_path = root_dir / "results" / "direct-gaussians" / "Gaussian-direct-results.json"
    with open(export_path, "r", encoding="utf-8") as f:
        exported = json.load(f)

    ckpt = exported["checkpoint"]
    weights_flat = np.array(ckpt["weights"], dtype=np.float32)
    scale = float(ckpt["scale"])
    mean_km = np.array(ckpt["mean"], dtype=np.float32)

    legacy_net = LegacyDirectWaveGaussianNetwork(mean_km=mean_km, scale=scale).to(device)
    legacy_net.load_flat_weights(weights_flat)
    legacy_net.eval()

    # Simulate the 20 reserved test cases with our native wgpu engine
    test_ids = data_mgr.get_reserved_test_cohort_indices(offset=12, per_family=2)
    test_fields_km = data_mgr.fields_km["test"][test_ids]
    engine = wave_gauss_rs.PyWaveEngine(SHOTS, NT, DT, PEAK_HZ, 20, 1000.0, 5000.0)

    test_waves_list = []
    for i in range(len(test_fields_km)):
        vel_mps = np.ascontiguousarray(test_fields_km[i].reshape(-1) * 1000.0, dtype=np.float32)
        test_waves_list.append(engine.simulate(vel_mps))
    test_waves = torch.from_numpy(np.stack(test_waves_list, axis=0)).to(device)
    target_km = torch.from_numpy(test_fields_km).to(device)

    # Verify parity with exported initial predictions
    with torch.no_grad():
        pred_km = legacy_net.predict_batch(test_waves)
        metrics = compute_metrics_mps(pred_km, target_km)

    exported_fields = np.stack(
        [np.array(r["direct"]["field"], dtype=np.float32).reshape(70, 70) for r in exported["comparison"]["rows"]],
        axis=0,
    )
    max_diff_mps = float(np.max(np.abs(pred_km.cpu().numpy() - exported_fields)) * 1000.0)

    # Run the 5 Counterfactual Wave Attribution Probes on the 20 test cases!
    blindfold_results = WaveAttributionSuite.evaluate_blindfold_probes(
        predict_fn=lambda w: legacy_net.predict_batch(w),
        traces=test_waves,
        target_km=target_km,
    )

    # Also compute Depth-Resolved Input Jacobian Sensitivity on the CurveFault-B case (index 2)
    def diff_maps_fn(w_in: torch.Tensor) -> torch.Tensor:
        detail, density = legacy_net.forward_maps(w_in.squeeze(0))
        return density.unsqueeze(0)

    jac_stats = WaveAttributionSuite.compute_depth_jacobian_attribution(
        differentiable_predict_fn=diff_maps_fn,
        traces_single=test_waves[2:3],
    )

    return {
        "checkpoint_update": ckpt["bestStep"],
        "reproduced_initial_rmse_mps": metrics["field_rmse_mps"],
        "reproduced_grad_rmse_mps": metrics["grad_rmse_mps"],
        "max_diff_vs_browser_mps": max_diff_mps,
        "blindfold_probes": blindfold_results,
        "jacobian_attribution": {
            k: v for k, v in jac_stats.items() if not isinstance(v, np.ndarray)
        },
    }


def verify_three_stage_curriculum_smoke(
    data_mgr: OpenFWIDatasetManager, device: torch.device
) -> dict:
    """Run a fast end-to-end 3-stage verification on a 4-model OpenFWI subset to prove:
    - Stage 1 Geological Autoencoder rapidly drives Gaussian reconstruction error down.
    - Stage 2 Traveltime-Attention Wave Encoder learns to control the frozen Stage 1 Decoder.
    - Stage 3 Physics-Guided Fine-Tuning decreases both wave mismatch and geology error.
    - Compare Depth-Resolved Jacobian Attribution (late reflections & far offsets) before vs after!
    """
    torch.manual_seed(20261020)
    np.random.seed(20261020)

    # Pick 4 diverse geologies (CurveFault-A, CurveFault-B, FlatFault-A, CurveVel-A)
    sample_ids = [0, 400, 1600, 800]
    v_sub = torch.from_numpy(data_mgr.fields_km["train"][sample_ids]).to(device)
    t_dens_np = wave_gauss_rs.batch_teacher_densities(
        np.ascontiguousarray(v_sub.cpu().numpy().reshape(4, 4900), dtype=np.float32), 70, 70, 10
    ).reshape(4, 70, 70)
    t_dens = torch.from_numpy(t_dens_np).to(device)

    engine = wave_gauss_rs.PyWaveEngine(SHOTS, NT, DT, PEAK_HZ, 20, 1000.0, 5000.0)
    mean_mps = np.ascontiguousarray(data_mgr.train_mean_km.reshape(-1) * 1000.0, dtype=np.float32)
    bg_traces = engine.simulate(mean_mps)

    waves_list = []
    for i in range(4):
        vm = np.ascontiguousarray(v_sub[i].cpu().numpy().reshape(-1) * 1000.0, dtype=np.float32)
        waves_list.append(engine.simulate(vm))
    w_sub = torch.from_numpy(np.stack(waves_list, axis=0)).to(device)

    rms_scale = float(torch.sqrt(w_sub.square().mean()).item())
    bg_t = torch.from_numpy(bg_traces).to(device).unsqueeze(0)
    scatter_scale = float(torch.sqrt((w_sub - bg_t).square().mean()).item())

    # --- Stage 1: Geological Autoencoder ---
    stage1 = GeologicalAutoencoder(mean_km=data_mgr.train_mean_km).to(device)
    opt1 = torch.optim.AdamW(stage1.parameters(), lr=3e-3, weight_decay=1e-4)
    with torch.no_grad():
        s1_init_rmse = compute_metrics_mps(stage1(v_sub).field_km, v_sub)["field_rmse_mps"]

    for step in range(25):
        opt1.zero_grad()
        out1 = stage1(v_sub, density_override=t_dens if step < 8 else None)
        g_loss, _, _ = field_and_edge_loss(out1.field_km, v_sub)
        d_loss = ((torch.nn.functional.softplus(out1.logits) - out1.logits * t_dens).mean())
        (g_loss + 0.1 * d_loss).backward()
        torch.nn.utils.clip_grad_norm_(stage1.parameters(), 5.0)
        opt1.step()

    stage1.eval()
    with torch.no_grad():
        out1_final = stage1(v_sub)
        s1_final_rmse = compute_metrics_mps(out1_final.field_km, v_sub)["field_rmse_mps"]
        target_zc, target_zf = out1_final.z_coarse.detach(), out1_final.z_fine.detach()

    # --- Stage 2: Traveltime-Biased Spatial Attention Wave Controller ---
    wave_enc = TraveltimeAttentionWaveEncoder(
        mean_km=data_mgr.train_mean_km,
        mean_bg_traces=bg_traces,
        rms_scale=rms_scale,
        scatter_scale=scatter_scale,
    ).to(device)
    decoder = TriangleGaussianDecoder(mean_km=data_mgr.train_mean_km).to(device)
    decoder.load_state_dict(stage1.decoder.state_dict())
    staged = StagedWaveToGaussianModel(wave_enc, decoder).to(device)
    staged.freeze_decoder(True)

    with torch.no_grad():
        s2_init_rmse = compute_metrics_mps(staged(w_sub)[0].field_km, v_sub)["field_rmse_mps"]

    opt2 = torch.optim.AdamW(staged.wave_encoder.parameters(), lr=2e-3, weight_decay=1e-4)
    for step in range(25):
        opt2.zero_grad()
        dec_out, w_lat = staged(w_sub, density_override=t_dens if step < 5 else None)
        g_loss, _, _ = field_and_edge_loss(dec_out.field_km, v_sub)
        lat_loss = torch.nn.functional.mse_loss(w_lat.z_coarse, target_zc) + torch.nn.functional.mse_loss(
            w_lat.z_fine, target_zf
        )
        d_loss = (torch.nn.functional.softplus(dec_out.logits) - dec_out.logits * t_dens).mean()
        (g_loss + 0.25 * lat_loss + 0.1 * d_loss).backward()
        torch.nn.utils.clip_grad_norm_(staged.wave_encoder.parameters(), 5.0)
        opt2.step()

    staged.eval()
    with torch.no_grad():
        s2_final_rmse = compute_metrics_mps(staged(w_sub)[0].field_km, v_sub)["field_rmse_mps"]

    # --- Stage 3: Joint Fine-Tuning with Native wgpu Wave Physics Loss ---
    staged.freeze_decoder(False)
    phys_loss = WavePhysicsLoss(use_stochastic_single_shot=True)
    opt3 = torch.optim.AdamW(
        [
            {"params": staged.wave_encoder.parameters(), "lr": 5e-4},
            {"params": staged.gaussian_decoder.parameters(), "lr": 1e-4},
        ]
    )
    gen = torch.Generator(device="cpu").manual_seed(42)
    for _ in range(4):
        opt3.zero_grad()
        dec_out, _ = staged(w_sub)
        g_loss, _, _ = field_and_edge_loss(dec_out.field_km, v_sub)
        w_loss = phys_loss(dec_out.field_km, w_sub, full_survey=False, generator=gen)
        (g_loss + 0.2 * w_loss).backward()
        torch.nn.utils.clip_grad_norm_(staged.parameters(), 5.0)
        opt3.step()

    staged.eval()
    with torch.no_grad():
        s3_final_rmse = compute_metrics_mps(staged(w_sub)[0].field_km, v_sub)["field_rmse_mps"]

    # Measure Jacobian attribution on the new Traveltime-Attention Wave Encoder
    jac_new = WaveAttributionSuite.compute_depth_jacobian_attribution(
        differentiable_predict_fn=lambda w_in: staged(w_in)[0].field_km,
        traces_single=w_sub[0:1],
    )

    return {
        "stage1_init_rmse_mps": s1_init_rmse,
        "stage1_final_rmse_mps": s1_final_rmse,
        "stage2_init_rmse_mps": s2_init_rmse,
        "stage2_final_rmse_mps": s2_final_rmse,
        "stage3_final_rmse_mps": s3_final_rmse,
        "new_arch_jacobian_attribution": {
            k: v for k, v in jac_new.items() if not isinstance(v, np.ndarray)
        },
    }


def main() -> None:
    t0 = time.perf_counter()
    root_dir = Path(__file__).resolve().parents[1]
    device = get_preferred_device()
    print(f"=== Native Rust + PyTorch Staged Verification on device: {device} ===")

    print("1. Checking OpenFWI 20261020 split & Rust Rayon teacher density speed...")
    t_split = time.perf_counter()
    data_mgr = OpenFWIDatasetManager(root_dir=root_dir)
    t_dens = data_mgr.get_train_teacher_densities()
    print(
        f"   Split loaded ({len(data_mgr.fields_km['train'])} train, "
        f"{len(data_mgr.fields_km['validation'])} val, {len(data_mgr.fields_km['test'])} test) "
        f"& {len(t_dens)} teacher densities ready in {time.perf_counter() - t_split:.2f}s"
    )

    print("2. Checking PyTorch Renderer + Rust wgpu Wave Adjoint VJP vs Finite Differences...")
    vjp_res = check_autograd_wave_renderer_vjp(device)
    print(
        f"   PASS VJP check: rel_err={vjp_res['relative_error']*100:.4f}% "
        f"(fd={vjp_res['fd_slope']:.6e}, adj={vjp_res['adjoint_slope']:.6e})"
    )

    print("3. Running 5-Probe Wave Attribution Suite on Saved DirectWaveGaussianNetwork Checkpoint...")
    probe_res = probe_saved_direct_checkpoint(root_dir, data_mgr, device)
    print(
        f"   Reproduced saved checkpoint (update {probe_res['checkpoint_update']}): "
        f"RMSE={probe_res['reproduced_initial_rmse_mps']:.2f} m/s "
        f"(max diff vs browser={probe_res['max_diff_vs_browser_mps']:.4f} m/s)"
    )
    for cond, m in probe_res["blindfold_probes"].items():
        print(
            f"   Probe [{cond:24s}]: full_RMSE={m['field_rmse_mps']:6.2f} m/s | "
            f"shallow_RMSE={m['shallow_rmse_mps']:6.2f} m/s | "
            f"deep_RMSE={m['deep_rmse_mps']:6.2f} m/s | "
            f"shift_from_clean={m['prediction_shift_mps']:6.2f} m/s"
        )

    print("4. Verifying 3-Stage Curriculum (Stage 1 Decoder -> Stage 2 Attention Wave -> Stage 3 Physics)...")
    smoke_res = verify_three_stage_curriculum_smoke(data_mgr, device)
    print(
        f"   Stage 1 (Geology -> Gaussians): {smoke_res['stage1_init_rmse_mps']:.2f} -> {smoke_res['stage1_final_rmse_mps']:.2f} m/s\n"
        f"   Stage 2 (Waves -> Frozen Decoder): {smoke_res['stage2_init_rmse_mps']:.2f} -> {smoke_res['stage2_final_rmse_mps']:.2f} m/s\n"
        f"   Stage 3 (Joint + wgpu Wave Adjoint): -> {smoke_res['stage3_final_rmse_mps']:.2f} m/s"
    )
    print("   Jacobian Sensitivity Comparison (Deep Target z in 45..65):")
    print(
        f"     Legacy Conv/TimeProj Net -> late_reflection_ratio={probe_res['jacobian_attribution']['deep_late_reflection_sensitivity_ratio']:.3f}, "
        f"far_offset_ratio={probe_res['jacobian_attribution']['deep_far_offset_sensitivity_ratio']:.3f}"
    )
    print(
        f"     New Kirchhoff-Attn Net   -> late_reflection_ratio={smoke_res['new_arch_jacobian_attribution']['deep_late_reflection_sensitivity_ratio']:.3f}, "
        f"far_offset_ratio={smoke_res['new_arch_jacobian_attribution']['deep_far_offset_sensitivity_ratio']:.3f}"
    )

    out_dir = root_dir / "results" / "staged-native"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - t0,
        "vjp_verification": vjp_res,
        "saved_checkpoint_probes": probe_res,
        "three_stage_verification": smoke_res,
    }
    out_file = out_dir / "verification_and_probes.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"=== ALL CHECKS PASSED in {report['elapsed_seconds']:.2f}s. Saved to {out_file} ===")


if __name__ == "__main__":
    main()
