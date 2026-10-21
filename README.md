# WGNet: Physics-Guided Staged Wave-to-Gaussian Seismic Inversion (Rust + PyTorch)

Physics-guided staged seismic velocity inversion and adaptive Delaunay Gaussian representation, powered by a cross-platform native **Rust (`PyO3` + `wgpu` + `rayon`)** core and **PyTorch (`mps` / `cuda`)** learning & attribution pipeline.

---

## Repository Structure

```text
WGNet/
├── Cargo.toml                  # Rust workspace manifest
├── crates/
│   ├── geology-core/           # Core 4th-order C-PML acoustic stencils, Ricker wavelet, and CPU reference solver
│   └── wave-gauss-rs/          # PyO3 native extension (`wave_gauss_rs`):
│       ├── src/lib.rs          # Zero-copy Python/NumPy bindings
│       ├── src/delaunay.rs     # Rayon-parallel serpentine Floyd–Steinberg dithering, Delaunay & Steiner priors
│       ├── src/wave_wgpu.rs    # Native GPU (Metal/Vulkan/DX12) multi-shot forward & exact discrete adjoint solver
│       └── src/shaders/        # WGSL compute kernels (`splatForward.wgsl`, `splatAdjoint.wgsl`)
├── python/
│   ├── pyproject.toml          # Python package configuration (`gaussian_lab`)
│   ├── verify_and_probe.py     # End-to-end physics gradient verification & counterfactual wave attribution suite
│   └── gaussian_lab/
│       ├── data.py             # OpenFWI (seed 20261020) split loader, teacher density cache & wgpu wave cache
│       ├── renderer.py         # Differentiable all-Gaussian normalized softmax renderer & structural edge losses
│       ├── stage1_decoder.py   # Stage 1: Geological Autoencoder & Triangle-Conditioned Gaussian Decoder
│       ├── stage2_wave_enc.py  # Stage 2: Physics-Guided Wave-to-Space Encoder & Staged Model
│       ├── stage3_physics.py   # Stage 3: Online wgpu Wave-Equation Adjoint `torch.autograd.Function`
│       ├── refine.py           # Test-time L-BFGS wave-equation Gaussian refinement
│       ├── probes.py           # 5-probe counterfactual wave blindfold suite & depth Jacobian attribution
│       └── train_staged.py     # Resumable 3-stage training & evaluation CLI
├── data/
│   ├── openfwi/                # OpenFWI 10-family velocity dataset (70x70, 10m spacing)
│   └── native_cache_.../       # Cached teacher densities & memory-mapped wgpu wavefields
├── results/
│   └── staged-native/          # Checkpoints (`stage{1,2,3}_checkpoint.pt`) & verification reports
└── legacy_code/                # Archived browser TypeScript/WebGPU/TF.js implementation & earlier results
```

---

## Quick Start Commands

```bash
# 1. Run Rust unit tests (Delaunay conservation, wgpu vs CPU parity, exact discrete adjoint vs FD)
cargo test --package wave-gauss-rs --release

# 2. Run end-to-end PyTorch + Rust verification and counterfactual wave attribution probes
./python/.venv/bin/python python/verify_and_probe.py

# 3. Train and evaluate the 3-stage curriculum (automatically resumes saved checkpoints)
./python/.venv/bin/python -m gaussian_lab.train_staged --stage all
```
