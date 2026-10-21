//! Scientific core in dependency-free Rust, compiled to WebAssembly.
//!
//! Stage 1 (Geology):
//! - OpenFWI 70 x 70 velocity model validation (10 m spacing, m/s)
//! - Reflected-boundary separable Gaussian smoothing G_sigma * v
//! - Structural velocity anomaly (v - G_sigma * v) and interface gradient |grad v|
//! - Deterministic 64-bit FNV-1a model hashing and Fisher-Yates disjoint split sampling
//!
//! Stage 2 (Simulate):
//! - Delayed Ricker source wavelet generation
//! - 4th-order finite-difference C-PML absorber profile generation (a, b, db/dx)
//! - Reproducible Box-Muller Gaussian recording noise (Clean vs 2% RMS noise) and trace metrics

pub const N: usize = 70;
pub const CELLS: usize = N * N;
pub const SPACING_M: f32 = 10.0;
pub const H: f32 = SPACING_M;
pub const V_MIN: f32 = 1500.0;
pub const V_MAX: f32 = 4500.0;
pub const FD_PAD: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModelSummary {
    pub min_v: f32,
    pub max_v: f32,
    pub mean_v: f32,
    pub std_v: f32,
    pub rms_anomaly: f32,
    pub rms_gradient: f32,
    pub max_gradient: f32,
    pub valid_bounds: bool,
    pub hash: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TraceSummary {
    pub clean_rms: f32,
    pub peak_abs: f32,
    pub noise_rms: f32,
    pub snr_db: f32,
}

/// Reflected boundary index on 0..N, matching RenderWave and src/geology.wgsl.
#[inline]
pub fn reflect(mut i: isize, n: usize) -> usize {
    let n = n as isize;
    while i < 0 || i >= n {
        i = if i < 0 { -i - 1 } else { 2 * n - i - 1 };
    }
    i as usize
}

/// Deterministic 64-bit FNV-1a hash of a 70 x 70 velocity field.
pub fn fnv1a_64(model: &[f32]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &v in model {
        let canonical = if v == 0.0 { 0.0f32 } else { v };
        for b in canonical.to_le_bytes() {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    hash
}

/// Separable Gaussian blur (sigma in grid nodes, radius = ceil(4 * sigma), reflected edges).
pub fn blur_model(src: &[f32], sigma: f32, out: &mut [f32]) {
    assert!(src.len() == CELLS && out.len() == CELLS);
    if sigma <= 0.0 {
        out.copy_from_slice(src);
        return;
    }
    let r = (4.0 * sigma).ceil() as isize;
    let mut weights = Vec::with_capacity((2 * r + 1) as usize);
    let mut sum = 0.0f32;
    for j in -r..=r {
        let u = (j as f32) / sigma;
        let w = (-0.5 * u * u).exp();
        weights.push(w);
        sum += w;
    }

    let mut temp = [0.0f32; CELLS];
    for z in 0..N {
        for x in 0..N {
            let mut acc = 0.0f32;
            for j in -r..=r {
                let xj = reflect(x as isize + j, N);
                acc += weights[(j + r) as usize] * src[z * N + xj];
            }
            temp[z * N + x] = acc / sum;
        }
    }
    for z in 0..N {
        for x in 0..N {
            let mut acc = 0.0f32;
            for j in -r..=r {
                let zj = reflect(z as isize + j, N);
                acc += weights[(j + r) as usize] * temp[zj * N + x];
            }
            out[z * N + x] = acc / sum;
        }
    }
}

/// Compute smooth background, velocity anomaly (v - blur), interface gradient |grad v|,
/// and summary statistics for one 70 x 70 velocity model.
pub fn analyze_model(
    v: &[f32],
    sigma: f32,
    blur_out: &mut [f32],
    anom_out: &mut [f32],
    grad_out: &mut [f32],
) -> ModelSummary {
    assert!(
        v.len() == CELLS
            && blur_out.len() == CELLS
            && anom_out.len() == CELLS
            && grad_out.len() == CELLS
    );
    blur_model(v, sigma, blur_out);

    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    let mut sum_v = 0.0f64;
    let mut sum_anom_sq = 0.0f64;
    let mut sum_grad_sq = 0.0f64;
    let mut max_grad = 0.0f32;
    let mut all_finite = true;

    for z in 0..N {
        let z0 = if z > 0 { z - 1 } else { 0 };
        let z1 = if z + 1 < N { z + 1 } else { N - 1 };
        let dz_scale = 1.0 / ((z1 - z0) as f32);

        for x in 0..N {
            let idx = z * N + x;
            let val = v[idx];
            if !val.is_finite() {
                all_finite = false;
            }
            if val < min_v {
                min_v = val;
            }
            if val > max_v {
                max_v = val;
            }
            sum_v += val as f64;

            let anom = val - blur_out[idx];
            anom_out[idx] = anom;
            sum_anom_sq += (anom as f64) * (anom as f64);

            let x0 = if x > 0 { x - 1 } else { 0 };
            let x1 = if x + 1 < N { x + 1 } else { N - 1 };
            let dx_scale = 1.0 / ((x1 - x0) as f32);

            let dv_dx = (v[z * N + x1] - v[z * N + x0]) * dx_scale;
            let dv_dz = (v[z1 * N + x] - v[z0 * N + x]) * dz_scale;
            let g = (dv_dx * dv_dx + dv_dz * dv_dz).sqrt();
            grad_out[idx] = g;
            sum_grad_sq += (g as f64) * (g as f64);
            if g > max_grad {
                max_grad = g;
            }
        }
    }

    let mean_v = (sum_v / (CELLS as f64)) as f32;
    let mut sum_var = 0.0f64;
    for &val in v {
        let d = (val - mean_v) as f64;
        sum_var += d * d;
    }
    let std_v = (sum_var / (CELLS as f64)).sqrt() as f32;
    let rms_anomaly = (sum_anom_sq / (CELLS as f64)).sqrt() as f32;
    let rms_gradient = (sum_grad_sq / (CELLS as f64)).sqrt() as f32;
    let valid_bounds = all_finite && min_v >= 1000.0 && max_v <= 6000.0;

    ModelSummary {
        min_v,
        max_v,
        mean_v,
        std_v,
        rms_anomaly,
        rms_gradient,
        max_gradient: max_grad,
        valid_bounds,
        hash: fnv1a_64(v),
    }
}

/// Mulberry32 seeded PRNG.
pub struct Rng(u32);
impl Rng {
    pub fn new(seed: u32) -> Self {
        Self(seed.wrapping_mul(2654435761).wrapping_add(0x9e3779b9))
    }
    pub fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_add(0x6d2b79f5);
        let mut t = self.0;
        t = (t ^ (t >> 15)).wrapping_mul(t | 1);
        t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
        t ^ (t >> 14)
    }
    pub fn next_f32_open01(&mut self) -> f32 {
        ((self.next_u32() >> 8) as f32 + 0.5) / 16_777_216.0
    }
    pub fn next_usize(&mut self, bound: usize) -> usize {
        if bound <= 1 {
            return 0;
        }
        (((self.next_u32() as u64) * (bound as u64)) >> 32) as usize
    }
    pub fn next_gaussian(&mut self) -> f32 {
        let u1 = self.next_f32_open01();
        let u2 = self.next_f32_open01();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

/// Deterministically sample `pick` unique model indices out of `0..total`.
pub fn sample_unique_indices(total: u32, pick: u32, seed: u32) -> Vec<u32> {
    let total_usize = total as usize;
    let pick_usize = (pick as usize).min(total_usize);
    let mut pool: Vec<u32> = (0..total).collect();
    let mut rng = Rng::new(seed);
    for i in 0..pick_usize {
        let j = i + rng.next_usize(total_usize - i);
        pool.swap(i, j);
    }
    pool.truncate(pick_usize);
    pool
}

// ---------------------------------------------------------------------------
// Stage 2 (Simulate): Ricker wavelet, C-PML profiles, and recording noise
// ---------------------------------------------------------------------------

/// Ricker wavelet delayed by 1.5 / peak_hz, matching RenderWave and deepwave.
pub fn ricker_wavelet(samples: usize, dt: f32, peak_hz: f32, out: &mut [f32]) {
    assert_eq!(out.len(), samples);
    let delay = 1.5 / peak_hz;
    for (i, slot) in out.iter_mut().enumerate() {
        let t = (i as f32) * dt - delay;
        let u = std::f32::consts::PI * peak_hz * t;
        let a = u * u;
        *slot = (1.0 - 2.0 * a) * (-a).exp();
    }
}

/// Deepwave's 4th-order C-PML profiles along one axis of `n` padded cells: a, b, and db/dx.
pub fn cpml_profiles(
    n: usize,
    h: f32,
    dt: f32,
    width: usize,
    max_vel: f32,
    freq: f32,
    a_out: &mut [f32],
    b_out: &mut [f32],
    db_out: &mut [f32],
) {
    assert!(a_out.len() == n && b_out.len() == n && db_out.len() == n);
    let mut a = vec![0.0f64; n];
    let mut b = vec![0.0f64; n];
    let start0 = (FD_PAD + width) as f64;
    let start1 = (n - 1 - FD_PAD - width) as f64;
    let width_f = width as f64;
    let h_f = h as f64;
    let dt_f = dt as f64;
    let sigma0 = -(3.0 * (max_vel as f64) * (0.001f64).ln()) / (2.0 * width_f * h_f);

    for x in 0..n {
        let xf = x as f64;
        let f0 = (start0 - xf) / width_f;
        let f1 = (xf - start1) / width_f;
        let frac = f0.max(f1).clamp(0.0, 1.0);
        let sigma = sigma0 * frac * frac;
        let alpha = std::f64::consts::PI * (freq as f64) * (1.0 - frac);
        let ax = (-(sigma + alpha) * dt_f).exp();
        b[x] = (sigma / (sigma + alpha)) * (ax - 1.0);
        a[x] = if frac == 0.0 { 0.0 } else { ax };
    }

    let mut db = vec![0.0f64; n];
    for x in 2..(n.saturating_sub(2)) {
        db[x] = ((8.0 / 12.0) * (b[x + 1] - b[x - 1]) - (1.0 / 12.0) * (b[x + 2] - b[x - 2])) / h_f;
    }

    for i in 0..n {
        a_out[i] = a[i] as f32;
        b_out[i] = b[i] as f32;
        db_out[i] = db[i] as f32;
    }
}

/// Add reproducible Gaussian noise to clean shot gathers:
///   d_obs = d_clean + N(0, (noise_fraction * rms(d_clean))^2)
/// Matches Gaussian vs Grid Lab's Clean (0.0) and 2% RMS noise (0.02) conditions.
pub fn apply_trace_noise(
    clean: &[f32],
    noise_fraction: f32,
    seed: u32,
    noisy_out: &mut [f32],
) -> TraceSummary {
    assert_eq!(clean.len(), noisy_out.len());
    let n = clean.len().max(1) as f64;
    let mut sum_sq = 0.0f64;
    let mut peak_abs = 0.0f32;
    for &c in clean {
        sum_sq += (c as f64) * (c as f64);
        let a = c.abs();
        if a > peak_abs {
            peak_abs = a;
        }
    }
    let clean_rms = (sum_sq / n).sqrt() as f32;
    if noise_fraction <= 0.0 || clean_rms == 0.0 {
        noisy_out.copy_from_slice(clean);
        return TraceSummary {
            clean_rms,
            peak_abs,
            noise_rms: 0.0,
            snr_db: f32::INFINITY,
        };
    }

    let sigma = clean_rms * noise_fraction;
    let mut rng = Rng::new(seed ^ 0x7303_2026);
    let mut noise_sq = 0.0f64;
    for (c, out) in clean.iter().zip(noisy_out.iter_mut()) {
        let w = rng.next_gaussian() * sigma;
        *out = *c + w;
        noise_sq += (w as f64) * (w as f64);
    }
    let noise_rms = (noise_sq / n).sqrt() as f32;
    let snr_db = if noise_rms > 0.0 {
        20.0 * (clean_rms / noise_rms).log10()
    } else {
        f32::INFINITY
    };
    TraceSummary {
        clean_rms,
        peak_abs,
        noise_rms,
        snr_db,
    }
}

/// CPU reference 4th-order finite-difference acoustic solver with C-PML for a single shot
/// (QuakeSim principle: the Rust CPU reference solver validates the WebGPU solver).
/// `v_vis` is the 70 x 70 visible velocity model; `out_traces` has length `nt * N`.
pub fn simulate_shot_cpu(
    v_vis: &[f32],
    h: f32,
    dt: f32,
    nt: usize,
    pml: usize,
    vmin: f32,
    vmax: f32,
    peak_hz: f32,
    src_x: f32,
    src_z: f32,
    out_traces: &mut [f32],
) {
    assert_eq!(v_vis.len(), CELLS);
    assert_eq!(out_traces.len(), nt * N);

    let pad = FD_PAD + pml;
    let nx = N + 2 * pad;
    let nz = N + 2 * pad;
    let cells = nx * nz;
    let region = pml + 2 * FD_PAD;
    let rh = 1.0f32 / h;
    let rh2 = rh * rh;
    let dt2 = dt * dt;

    // Padded velocity grid (clamped to visible edges and [vmin, vmax], matching splatKernels.wgsl)
    let mut vel = vec![0.0f32; cells];
    for z in 0..nz {
        let vz = (z as isize - pad as isize).clamp(0, (N - 1) as isize) as usize;
        for x in 0..nx {
            let vx = (x as isize - pad as isize).clamp(0, (N - 1) as isize) as usize;
            vel[z * nx + x] = v_vis[vz * N + vx].clamp(vmin, vmax);
        }
    }

    let (mut ax, mut bx, mut dbx) = (vec![0.0f32; nx], vec![0.0f32; nx], vec![0.0f32; nx]);
    let (mut az, mut bz, mut dbz) = (vec![0.0f32; nz], vec![0.0f32; nz], vec![0.0f32; nz]);
    cpml_profiles(nx, h, dt, pml, vmax, peak_hz, &mut ax, &mut bx, &mut dbx);
    cpml_profiles(nz, h, dt, pml, vmax, peak_hz, &mut az, &mut bz, &mut dbz);

    let mut wavelet = vec![0.0f32; nt];
    ricker_wavelet(nt, dt, peak_hz, &mut wavelet);

    // Bilinear corners for the source on the padded grid
    let gx = src_x / h + (pad as f32);
    let gz = src_z / h + (pad as f32);
    let x0 = (gx.floor() as usize).min(nx - 2);
    let z0 = (gz.floor() as usize).min(nz - 2);
    let fx = gx - (x0 as f32);
    let fz = gz - (z0 as f32);
    let src_corners = [
        (z0 * nx + x0, (1.0 - fx) * (1.0 - fz)),
        (z0 * nx + x0 + 1, fx * (1.0 - fz)),
        ((z0 + 1) * nx + x0, (1.0 - fx) * fz),
        ((z0 + 1) * nx + x0 + 1, fx * fz),
    ];

    // Surface receivers at x = r * h, z = h -> padded grid node (pad + r, pad + 1)
    let rec_z = pad + 1;

    let mut cur = vec![0.0f32; cells];
    let mut prev = vec![0.0f32; cells];
    let mut psi_x = vec![0.0f32; cells];
    let mut psi_x_next = vec![0.0f32; cells];
    let mut psi_z = vec![0.0f32; cells];
    let mut psi_z_next = vec![0.0f32; cells];
    let mut zeta_x = vec![0.0f32; cells];
    let mut zeta_z = vec![0.0f32; cells];

    const C1: f32 = 0.6666666666666666;
    const C2: f32 = 0.08333333333333333;
    const D0: f32 = -2.5;
    const D1: f32 = 1.3333333333333333;
    const D2: f32 = 0.08333333333333333;

    for t in 0..nt {
        // Record surface receivers before advancing step t
        let row = &mut out_traces[t * N..(t + 1) * N];
        for r in 0..N {
            row[r] = cur[rec_z * nx + (pad + r)];
        }

        for z in FD_PAD..(nz - FD_PAD) {
            let in_pml_z = z < region || z >= nz - region;
            for x in FD_PAD..(nx - FD_PAD) {
                let i = z * nx + x;
                let u = cur[i];
                let mut w = 0.0f32;

                // x-axis 4th-order derivatives and C-PML memory variables
                let um1x = cur[i - 1];
                let up1x = cur[i + 1];
                let um2x = cur[i - 2];
                let up2x = cur[i + 2];
                let d2x = (D0 * u + D1 * (up1x + um1x) - D2 * (up2x + um2x)) * rh2;
                if x < region || x >= nx - region {
                    let d1x = (C1 * (up1x - um1x) - C2 * (up2x - um2x)) * rh;
                    let a = ax[x];
                    let b = bx[x];
                    let ap1 = ax[x + 1] * psi_x[i + 1];
                    let am1 = ax[x - 1] * psi_x[i - 1];
                    let ap2 = ax[x + 2] * psi_x[i + 2];
                    let am2 = ax[x - 2] * psi_x[i - 2];
                    let d1_apsi = (C1 * (ap1 - am1) - C2 * (ap2 - am2)) * rh;
                    let tmp = (1.0 + b) * d2x + dbx[x] * d1x + d1_apsi;
                    w += (1.0 + b) * tmp + a * zeta_x[i];
                    psi_x_next[i] = b * d1x + a * psi_x[i];
                    zeta_x[i] = b * tmp + a * zeta_x[i];
                } else {
                    w += d2x;
                    psi_x_next[i] = 0.0;
                }

                // z-axis 4th-order derivatives and C-PML memory variables
                let um1z = cur[i - nx];
                let up1z = cur[i + nx];
                let um2z = cur[i - 2 * nx];
                let up2z = cur[i + 2 * nx];
                let d2z = (D0 * u + D1 * (up1z + um1z) - D2 * (up2z + um2z)) * rh2;
                if in_pml_z {
                    let d1z = (C1 * (up1z - um1z) - C2 * (up2z - um2z)) * rh;
                    let a = az[z];
                    let b = bz[z];
                    let ap1 = az[z + 1] * psi_z[i + nx];
                    let am1 = az[z - 1] * psi_z[i - nx];
                    let ap2 = az[z + 2] * psi_z[i + 2 * nx];
                    let am2 = az[z - 2] * psi_z[i - 2 * nx];
                    let d1_apsi = (C1 * (ap1 - am1) - C2 * (ap2 - am2)) * rh;
                    let tmp = (1.0 + b) * d2z + dbz[z] * d1z + d1_apsi;
                    w += (1.0 + b) * tmp + a * zeta_z[i];
                    psi_z_next[i] = b * d1z + a * psi_z[i];
                    zeta_z[i] = b * tmp + a * zeta_z[i];
                } else {
                    w += d2z;
                    psi_z_next[i] = 0.0;
                }

                let v = vel[i];
                let v2dt2 = v * v * dt2;
                prev[i] = v2dt2 * w + 2.0 * u - prev[i];
            }
        }

        // Inject point source into the 4 bilinear corner cells
        let f_t = wavelet[t];
        for &(cell, weight) in &src_corners {
            if weight != 0.0 {
                let v = vel[cell];
                prev[cell] += -(v * v * dt2) * weight * f_t;
            }
        }

        std::mem::swap(&mut cur, &mut prev);
        std::mem::swap(&mut psi_x, &mut psi_x_next);
        std::mem::swap(&mut psi_z, &mut psi_z_next);
    }
}

// ---------------------------------------------------------------------------
// Stage 3 (Train): Gaussian-only starting-model prior from seismic recordings
// ---------------------------------------------------------------------------

pub const FEAT_DIM: usize = 128; // 8 time windows x 8 receiver bins x 2 channels (signed mean + RMS envelope)
pub const HIDDEN_DIM: usize = 32;
pub const SLOTS: usize = 8;      // [amp, x, z, log_sx, log_sz, theta, e1, e2] matching SplatFwi
pub const GAUSS_HEAD: usize = 6; // learned offsets per Gaussian: [amp, dx, dz, dlog_sx, dlog_sz, theta]
const CUT: f32 = 25.0;
const EDGE_CUT: f32 = 3.726653172078671e-6; // exp(-12.5)
const XMAX: f32 = ((N - 1) as f32) * H;     // 690.0 m

/// Number of learnable weights in the Gaussian prior network for a k_side x k_side anchor lattice.
pub const fn prior_param_count(k_side: usize) -> usize {
    let k = k_side * k_side;
    let out_dim = k * GAUSS_HEAD;
    FEAT_DIM * HIDDEN_DIM + HIDDEN_DIM + HIDDEN_DIM * out_dim + out_dim
}

/// Extract 128 compact recording features from (shots, nt, nrec) traces using asinh(trace / scale),
/// with optional 0..noise_max independent Gaussian recording noise (matching prior.py).
pub fn extract_recording_features(
    traces: &[f32],
    shots: usize,
    nt: usize,
    nrec: usize,
    input_scale: f32,
    noise_frac: f32,
    seed: u32,
    out_features: &mut [f32],
) {
    assert_eq!(traces.len(), shots * nt * nrec);
    assert_eq!(out_features.len(), FEAT_DIM);

    let scale = input_scale.max(1e-6);
    let mut rng = Rng::new(seed);
    let noise_std = if noise_frac > 0.0 {
        let mut sum_sq = 0.0f64;
        for &v in traces {
            sum_sq += (v as f64) * (v as f64);
        }
        let rms = (sum_sq / (traces.len().max(1) as f64)).sqrt() as f32;
        rms * noise_frac
    } else {
        0.0
    };

    out_features.fill(0.0);
    let mut counts = [0u32; 64];

    for s in 0..shots {
        for t in 0..nt {
            let tb = ((t * 8) / nt.max(1)).min(7);
            let row_offset = (s * nt + t) * nrec;
            for r in 0..nrec {
                let rb = ((r * 8) / nrec.max(1)).min(7);
                let bin = tb * 8 + rb;
                let noisy = if noise_std > 0.0 {
                    traces[row_offset + r] + noise_std * rng.next_gaussian()
                } else {
                    traces[row_offset + r]
                };
                // asinh(x / scale) compression as in WaveInitializer.forward
                let u = (noisy / scale).asinh();
                out_features[bin] += u;
                out_features[64 + bin] += u * u;
                counts[bin] += 1;
            }
        }
    }

    for bin in 0..64 {
        let c = (counts[bin].max(1)) as f32;
        out_features[bin] /= c;
        out_features[64 + bin] = (out_features[64 + bin] / c).sqrt();
    }
}

/// Initialize the Gaussian prior weights: Xavier normal on layer 1 (encoder), zeros on layer 2 (Gaussian head)
/// so Epoch 0 outputs zero-amplitude Gaussians on top of the mean training background (matching core/models.py).
pub fn init_gaussian_prior(k_side: usize, seed: u32, params: &mut [f32]) {
    assert_eq!(params.len(), prior_param_count(k_side));
    params.fill(0.0);
    let mut rng = Rng::new(seed);
    let scale = (2.0 / ((FEAT_DIM + HIDDEN_DIM) as f32)).sqrt();
    for w in &mut params[0..(FEAT_DIM * HIDDEN_DIM)] {
        *w = scale * rng.next_gaussian();
    }
}

/// Differentiable Gaussian splatting on the 70 x 70 grid + (Field MSE + 0.3 * Edge MSE) loss and exact gradient
/// with respect to `gaussians` (count * 8 floats). Loss is scaled in (km/s)^2 to match Gaussian Wave Lab.
pub fn splat_and_loss_grad(
    base_v: &[f32],
    target_v: &[f32],
    gaussians: &[f32],
    pred_v: &mut [f32],
    grad_gaussians: &mut [f32],
    compute_grad: bool,
) -> (f32, f32, f32) {
    assert_eq!(base_v.len(), CELLS);
    assert_eq!(target_v.len(), CELLS);
    assert_eq!(pred_v.len(), CELLS);
    let count = gaussians.len() / SLOTS;
    assert_eq!(gaussians.len(), count * SLOTS);
    if compute_grad {
        assert_eq!(grad_gaussians.len(), count * SLOTS);
        grad_gaussians.fill(0.0);
    }

    let mut unclamped = [0.0f32; CELLS];
    unclamped.copy_from_slice(base_v);

    // 1. Splat all K Gaussians onto the 70 x 70 visible grid
    for k in 0..count {
        let g = k * SLOTS;
        let amp = gaussians[g];
        if amp == 0.0 && !compute_grad {
            continue;
        }
        let cx = gaussians[g + 1];
        let cz = gaussians[g + 2];
        let sx = gaussians[g + 3].exp().max(5.0);
        let sz = gaussians[g + 4].exp().max(5.0);
        let c = gaussians[g + 5].cos();
        let s = gaussians[g + 5].sin();

        let ex = 5.0 * (sx * sx * c * c + sz * sz * s * s).sqrt();
        let ez = 5.0 * (sx * sx * s * s + sz * sz * c * c).sqrt();
        let x0 = (((cx - ex) / H).floor() as isize).clamp(0, (N - 1) as isize) as usize;
        let x1 = (((cx + ex) / H).ceil() as isize).clamp(0, (N - 1) as isize) as usize;
        let z0 = (((cz - ez) / H).floor() as isize).clamp(0, (N - 1) as isize) as usize;
        let z1 = (((cz + ez) / H).ceil() as isize).clamp(0, (N - 1) as isize) as usize;

        for z in z0..=z1 {
            let dz = (z as f32) * H - cz;
            for x in x0..=x1 {
                let dx = (x as f32) * H - cx;
                let u1 = (c * dx + s * dz) / sx;
                let u2 = (-s * dx + c * dz) / sz;
                let q = u1 * u1 + u2 * u2;
                if q < CUT {
                    unclamped[z * N + x] += amp * ((-0.5 * q).exp() - EDGE_CUT);
                }
            }
        }
    }

    let mut mask = [1.0f32; CELLS];
    for i in 0..CELLS {
        let v = unclamped[i];
        if v < 1000.0 {
            pred_v[i] = 1000.0;
            mask[i] = 0.0;
        } else if v > 5000.0 {
            pred_v[i] = 5000.0;
            mask[i] = 0.0;
        } else {
            pred_v[i] = v;
        }
    }

    // 2. Compute Field MSE + 0.3 * Boundary-Difference MSE (in km/s units for well-scaled gradients)
    let mut err = [0.0f32; CELLS];
    let mut mse_sum = 0.0f64;
    for i in 0..CELLS {
        let e = (pred_v[i] - target_v[i]) * 0.001; // km/s
        err[i] = e;
        mse_sum += (e as f64) * (e as f64);
    }
    let mse = (mse_sum / (CELLS as f64)) as f32;
    let rmse_mps = (mse.sqrt()) * 1000.0;

    let edge_pairs = (N * (N - 1) + (N - 1) * N) as f64;
    let mut edge_sum = 0.0f64;
    for z in 0..N {
        for x in 0..(N - 1) {
            let de = (err[z * N + x + 1] - err[z * N + x]) as f64;
            edge_sum += de * de;
        }
    }
    for z in 0..(N - 1) {
        for x in 0..N {
            let de = (err[(z + 1) * N + x] - err[z * N + x]) as f64;
            edge_sum += de * de;
        }
    }
    let edge_mse = (edge_sum / edge_pairs) as f32;
    let edge_rmse_mps = (edge_mse.sqrt()) * 1000.0;
    let loss = mse + 0.3 * edge_mse;

    if !compute_grad {
        return (loss, rmse_mps, edge_rmse_mps);
    }

    // 3. Exact pixel gradient dL / d(pred_v[i]) (in m/s^-1)
    let mut grad_v = [0.0f32; CELLS];
    let scale_mse = (2.0 / (CELLS as f32)) * 1e-3;
    let scale_edge = 0.3 * (2.0 / (edge_pairs as f32)) * 1e-3;

    for i in 0..CELLS {
        grad_v[i] = scale_mse * err[i];
    }
    for z in 0..N {
        for x in 0..(N - 1) {
            let i0 = z * N + x;
            let i1 = i0 + 1;
            let g = scale_edge * (err[i1] - err[i0]);
            grad_v[i1] += g;
            grad_v[i0] -= g;
        }
    }
    for z in 0..(N - 1) {
        for x in 0..N {
            let i0 = z * N + x;
            let i1 = i0 + N;
            let g = scale_edge * (err[i1] - err[i0]);
            grad_v[i1] += g;
            grad_v[i0] -= g;
        }
    }
    for i in 0..CELLS {
        grad_v[i] *= mask[i];
    }

    // 4. Exact Gaussian splat backward pass (matching splatKernels.wgsl::backward)
    for k in 0..count {
        let g = k * SLOTS;
        let amp = gaussians[g];
        let cx = gaussians[g + 1];
        let cz = gaussians[g + 2];
        let sx = gaussians[g + 3].exp().max(5.0);
        let sz = gaussians[g + 4].exp().max(5.0);
        let c = gaussians[g + 5].cos();
        let s = gaussians[g + 5].sin();

        let ex = 5.0 * (sx * sx * c * c + sz * sz * s * s).sqrt();
        let ez = 5.0 * (sx * sx * s * s + sz * sz * c * c).sqrt();
        let x0 = (((cx - ex) / H).floor() as isize).clamp(0, (N - 1) as isize) as usize;
        let x1 = (((cx + ex) / H).ceil() as isize).clamp(0, (N - 1) as isize) as usize;
        let z0 = (((cz - ez) / H).floor() as isize).clamp(0, (N - 1) as isize) as usize;
        let z1 = (((cz + ez) / H).ceil() as isize).clamp(0, (N - 1) as isize) as usize;

        let mut acc = [0.0f32; 6];
        for z in z0..=z1 {
            let dz = (z as f32) * H - cz;
            for x in x0..=x1 {
                let dx = (x as f32) * H - cx;
                let r1 = c * dx + s * dz;
                let r2 = -s * dx + c * dz;
                let u1 = r1 / sx;
                let u2 = r2 / sz;
                let q = u1 * u1 + u2 * u2;
                if q < CUT {
                    let gv = grad_v[z * N + x];
                    let big_g = (-0.5 * q).exp();
                    let ag = gv * amp * big_g;
                    acc[0] += gv * (big_g - EDGE_CUT);
                    acc[1] += ag * (u1 / sx * c - u2 / sz * s);
                    acc[2] += ag * (u1 / sx * s + u2 / sz * c);
                    acc[3] += ag * u1 * u1;
                    acc[4] += ag * u2 * u2;
                    acc[5] += ag * (-r1 * r2 * (1.0 / (sx * sx) - 1.0 / (sz * sz)));
                }
            }
        }
        grad_gaussians[g..g + 6].copy_from_slice(&acc);
    }

    (loss, rmse_mps, edge_rmse_mps)
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn silu_and_deriv(x: f32) -> (f32, f32) {
    let s = sigmoid(x);
    (x * s, s * (1.0 + x * (1.0 - s)))
}

/// Forward pass of the Gaussian prior network: 128 recording features -> 32 SiLU hidden -> K x 8 Gaussians.
pub fn forward_gaussian_prior(
    features: &[f32],
    k_side: usize,
    params: &[f32],
    h1_out: &mut [f32],
    dh1_out: &mut [f32],
    tanh_out: &mut [f32],
    gaussians_out: &mut [f32],
) {
    let k_total = k_side * k_side;
    let out_dim = k_total * GAUSS_HEAD;
    assert_eq!(features.len(), FEAT_DIM);
    assert_eq!(params.len(), prior_param_count(k_side));
    assert_eq!(h1_out.len(), HIDDEN_DIM);
    assert_eq!(dh1_out.len(), HIDDEN_DIM);
    assert_eq!(tanh_out.len(), out_dim);
    assert_eq!(gaussians_out.len(), k_total * SLOTS);

    let w1_end = FEAT_DIM * HIDDEN_DIM;
    let b1_end = w1_end + HIDDEN_DIM;
    let w2_end = b1_end + HIDDEN_DIM * out_dim;

    let w1 = &params[0..w1_end];
    let b1 = &params[w1_end..b1_end];
    let w2 = &params[b1_end..w2_end];
    let b2 = &params[w2_end..];

    for h in 0..HIDDEN_DIM {
        let mut z = b1[h];
        for f in 0..FEAT_DIM {
            z += features[f] * w1[f * HIDDEN_DIM + h];
        }
        let (act, deriv) = silu_and_deriv(z);
        h1_out[h] = act;
        dh1_out[h] = deriv;
    }

    let base_log_w = ((0.65 * XMAX) / (k_side as f32)).ln();
    let max_shift = (1.5 * XMAX) / (k_side as f32);
    let scales = [1500.0f32, max_shift, max_shift, 0.75, 0.75, 1.2];

    for k in 0..k_total {
        let row = k / k_side;
        let col = k % k_side;
        let ax = ((col + 1) as f32 * XMAX) / ((k_side + 1) as f32);
        let az = ((row + 1) as f32 * XMAX) / ((k_side + 1) as f32);
        let anchors = [0.0f32, ax, az, base_log_w, base_log_w, 0.0];

        let g = k * SLOTS;
        for p in 0..GAUSS_HEAD {
            let j = k * GAUSS_HEAD + p;
            let mut u = b2[j];
            for h in 0..HIDDEN_DIM {
                u += h1_out[h] * w2[h * out_dim + j];
            }
            let t = u.tanh();
            tanh_out[j] = t;
            gaussians_out[g + p] = anchors[p] + scales[p] * t;
        }
        gaussians_out[g + 6] = 0.0;
        gaussians_out[g + 7] = 0.0;
    }
}

/// Predict the Gaussian set (K * 8) and splatted 70 x 70 velocity field from recording features,
/// and evaluate (loss, rmse_mps, edge_rmse_mps) against `target_v`.
pub fn predict_gaussian_prior(
    features: &[f32],
    base_v: &[f32],
    target_v: &[f32],
    k_side: usize,
    params: &[f32],
    pred_v: &mut [f32],
    gaussians_out: &mut [f32],
) -> (f32, f32, f32) {
    let k_total = k_side * k_side;
    let out_dim = k_total * GAUSS_HEAD;
    let mut h1 = [0.0f32; HIDDEN_DIM];
    let mut dh1 = [0.0f32; HIDDEN_DIM];
    let mut tanh_buf = vec![0.0f32; out_dim];
    forward_gaussian_prior(features, k_side, params, &mut h1, &mut dh1, &mut tanh_buf, gaussians_out);
    splat_and_loss_grad(base_v, target_v, gaussians_out, pred_v, &mut [], false)
}

/// One training step of the Gaussian prior network (forward, splat loss + analytical backward,
/// global gradient clip at 1.0, and AdamW update).
pub fn train_gaussian_prior_step(
    features: &[f32],
    base_v: &[f32],
    target_v: &[f32],
    k_side: usize,
    step_idx: u32,
    lr: f32,
    weight_decay: f32,
    params: &mut [f32],
    m1: &mut [f32],
    m2: &mut [f32],
    pred_v: &mut [f32],
    gaussians_out: &mut [f32],
) -> (f32, f32, f32) {
    let k_total = k_side * k_side;
    let out_dim = k_total * GAUSS_HEAD;
    let total_params = prior_param_count(k_side);
    assert_eq!(params.len(), total_params);
    assert_eq!(m1.len(), total_params);
    assert_eq!(m2.len(), total_params);

    let mut h1 = [0.0f32; HIDDEN_DIM];
    let mut dh1 = [0.0f32; HIDDEN_DIM];
    let mut tanh_buf = vec![0.0f32; out_dim];
    forward_gaussian_prior(features, k_side, params, &mut h1, &mut dh1, &mut tanh_buf, gaussians_out);

    let mut grad_gauss = vec![0.0f32; k_total * SLOTS];
    let (loss, rmse_mps, edge_rmse_mps) =
        splat_and_loss_grad(base_v, target_v, gaussians_out, pred_v, &mut grad_gauss, true);

    // Backpropagate through tanh bounds to u = W2 h1 + b2
    let max_shift = (1.5 * XMAX) / (k_side as f32);
    let scales = [1500.0f32, max_shift, max_shift, 0.75, 0.75, 1.2];
    let mut du = vec![0.0f32; out_dim];
    for k in 0..k_total {
        let g = k * SLOTS;
        for p in 0..GAUSS_HEAD {
            let j = k * GAUSS_HEAD + p;
            let t = tanh_buf[j];
            du[j] = grad_gauss[g + p] * scales[p] * (1.0 - t * t);
        }
    }

    let w1_end = FEAT_DIM * HIDDEN_DIM;
    let b1_end = w1_end + HIDDEN_DIM;
    let w2_end = b1_end + HIDDEN_DIM * out_dim;

    let mut grads = vec![0.0f32; total_params];
    let mut grad_h1 = [0.0f32; HIDDEN_DIM];

    // Layer 2 gradients (w2, b2) and backprop to h1
    for h in 0..HIDDEN_DIM {
        let act = h1[h];
        let w2_row = b1_end + h * out_dim;
        let mut acc_h = 0.0f32;
        for j in 0..out_dim {
            let dj = du[j];
            grads[w2_row + j] = act * dj;
            acc_h += params[w2_row + j] * dj;
        }
        grad_h1[h] = acc_h;
    }
    grads[w2_end..].copy_from_slice(&du);

    // Layer 1 gradients (w1, b1) through SiLU derivative
    for h in 0..HIDDEN_DIM {
        let dz = grad_h1[h] * dh1[h];
        grads[w1_end + h] = dz;
        for f in 0..FEAT_DIM {
            grads[f * HIDDEN_DIM + h] = features[f] * dz;
        }
    }

    // Global gradient norm clip at 1.0 (matching prior.py: clip_grad_norm_(model.parameters(), 1.))
    let mut norm_sq = 0.0f64;
    for &g in &grads {
        norm_sq += (g as f64) * (g as f64);
    }
    let norm = norm_sq.sqrt() as f32;
    let clip_scale = if norm > 1.0 { 1.0 / norm } else { 1.0 };

    // AdamW parameter update
    let beta1 = 0.9f32;
    let beta2 = 0.999f32;
    let step_f = (step_idx.max(1)) as i32;
    let bc1 = 1.0 - beta1.powi(step_f);
    let bc2 = 1.0 - beta2.powi(step_f);

    for i in 0..total_params {
        let g = grads[i] * clip_scale;
        m1[i] = beta1 * m1[i] + (1.0 - beta1) * g;
        m2[i] = beta2 * m2[i] + (1.0 - beta2) * g * g;
        let m_hat = m1[i] / bc1;
        let v_hat = m2[i] / bc2;
        params[i] = params[i] * (1.0 - lr * weight_decay) - lr * m_hat / (v_hat.sqrt() + 1e-8);
    }

    (loss, rmse_mps, edge_rmse_mps)
}

// ---------------------------------------------------------------------------
// WebAssembly C ABI exports
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn alloc_f32(len: usize) -> *mut f32 {
    let mut buf = vec![0.0f32; len];
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

/// # Safety
/// `ptr` and `len` must come from `alloc_f32`.
#[no_mangle]
pub unsafe extern "C" fn free_f32(ptr: *mut f32, len: usize) {
    if !ptr.is_null() && len > 0 {
        drop(Vec::from_raw_parts(ptr, len, len));
    }
}

#[no_mangle]
pub extern "C" fn alloc_u32(len: usize) -> *mut u32 {
    let mut buf = vec![0u32; len];
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

/// # Safety
/// `ptr` and `len` must come from `alloc_u32`.
#[no_mangle]
pub unsafe extern "C" fn free_u32(ptr: *mut u32, len: usize) {
    if !ptr.is_null() && len > 0 {
        drop(Vec::from_raw_parts(ptr, len, len));
    }
}

/// # Safety
/// All pointers must reference valid slices of length `CELLS` (4900), except `stats_ptr` (length 8).
#[no_mangle]
pub unsafe extern "C" fn wasm_analyze_model(
    v_ptr: *const f32,
    sigma: f32,
    blur_ptr: *mut f32,
    anom_ptr: *mut f32,
    grad_ptr: *mut f32,
    stats_ptr: *mut f32,
) -> u64 {
    let v = std::slice::from_raw_parts(v_ptr, CELLS);
    let blur = std::slice::from_raw_parts_mut(blur_ptr, CELLS);
    let anom = std::slice::from_raw_parts_mut(anom_ptr, CELLS);
    let grad = std::slice::from_raw_parts_mut(grad_ptr, CELLS);
    let stats = std::slice::from_raw_parts_mut(stats_ptr, 8);

    let s = analyze_model(v, sigma, blur, anom, grad);
    stats[0] = s.min_v;
    stats[1] = s.max_v;
    stats[2] = s.mean_v;
    stats[3] = s.std_v;
    stats[4] = s.rms_anomaly;
    stats[5] = s.rms_gradient;
    stats[6] = s.max_gradient;
    stats[7] = if s.valid_bounds { 1.0 } else { 0.0 };
    s.hash
}

/// # Safety
/// `v_ptr` must reference `CELLS` (4900) floats.
#[no_mangle]
pub unsafe extern "C" fn wasm_model_hash(v_ptr: *const f32) -> u64 {
    let v = std::slice::from_raw_parts(v_ptr, CELLS);
    fnv1a_64(v)
}

/// # Safety
/// `out_ptr` must have capacity for `pick.min(total)` u32 values.
#[no_mangle]
pub unsafe extern "C" fn wasm_sample_indices(
    total: u32,
    pick: u32,
    seed: u32,
    out_ptr: *mut u32,
) {
    let indices = sample_unique_indices(total, pick, seed);
    let out = std::slice::from_raw_parts_mut(out_ptr, indices.len());
    out.copy_from_slice(&indices);
}

/// # Safety
/// `out_ptr` must reference `samples` floats.
#[no_mangle]
pub unsafe extern "C" fn wasm_ricker_wavelet(
    samples: usize,
    dt: f32,
    peak_hz: f32,
    out_ptr: *mut f32,
) {
    let out = std::slice::from_raw_parts_mut(out_ptr, samples);
    ricker_wavelet(samples, dt, peak_hz, out);
}

/// # Safety
/// `out_ptr` must reference `3 * n` floats (a, then b, then db).
#[no_mangle]
pub unsafe extern "C" fn wasm_cpml_profiles(
    n: usize,
    h: f32,
    dt: f32,
    width: usize,
    max_vel: f32,
    freq: f32,
    out_ptr: *mut f32,
) {
    let all = std::slice::from_raw_parts_mut(out_ptr, 3 * n);
    let (a, rest) = all.split_at_mut(n);
    let (b, db) = rest.split_at_mut(n);
    cpml_profiles(n, h, dt, width, max_vel, freq, a, b, db);
}

/// # Safety
/// `clean_ptr` and `noisy_ptr` must reference `len` floats; `stats_ptr` must reference 4 floats.
#[no_mangle]
pub unsafe extern "C" fn wasm_apply_trace_noise(
    clean_ptr: *const f32,
    len: usize,
    noise_fraction: f32,
    seed: u32,
    noisy_ptr: *mut f32,
    stats_ptr: *mut f32,
) {
    let clean = std::slice::from_raw_parts(clean_ptr, len);
    let noisy = std::slice::from_raw_parts_mut(noisy_ptr, len);
    let stats = std::slice::from_raw_parts_mut(stats_ptr, 4);
    let s = apply_trace_noise(clean, noise_fraction, seed, noisy);
    stats[0] = s.clean_rms;
    stats[1] = s.peak_abs;
    stats[2] = s.noise_rms;
    stats[3] = s.snr_db;
}

/// # Safety
/// `v_ptr` must reference `CELLS` (4900) floats; `out_ptr` must reference `nt * N` floats.
#[no_mangle]
pub unsafe extern "C" fn wasm_simulate_shot_cpu(
    v_ptr: *const f32,
    h: f32,
    dt: f32,
    nt: usize,
    pml: usize,
    vmin: f32,
    vmax: f32,
    peak_hz: f32,
    src_x: f32,
    src_z: f32,
    out_ptr: *mut f32,
) {
    let v = std::slice::from_raw_parts(v_ptr, CELLS);
    let out = std::slice::from_raw_parts_mut(out_ptr, nt * N);
    simulate_shot_cpu(v, h, dt, nt, pml, vmin, vmax, peak_hz, src_x, src_z, out);
}

#[no_mangle]
pub extern "C" fn wasm_prior_param_count(k_side: usize) -> usize {
    prior_param_count(k_side)
}

/// # Safety
/// `params_ptr` must reference `prior_param_count(k_side)` floats.
#[no_mangle]
pub unsafe extern "C" fn wasm_init_gaussian_prior(k_side: usize, seed: u32, params_ptr: *mut f32) {
    let len = prior_param_count(k_side);
    let params = std::slice::from_raw_parts_mut(params_ptr, len);
    init_gaussian_prior(k_side, seed, params);
}

/// # Safety
/// `traces_ptr` must reference `shots * nt * nrec` floats; `out_ptr` must reference `FEAT_DIM` (128) floats.
#[no_mangle]
pub unsafe extern "C" fn wasm_extract_recording_features(
    traces_ptr: *const f32,
    shots: usize,
    nt: usize,
    nrec: usize,
    input_scale: f32,
    noise_frac: f32,
    seed: u32,
    out_ptr: *mut f32,
) {
    let traces = std::slice::from_raw_parts(traces_ptr, shots * nt * nrec);
    let out = std::slice::from_raw_parts_mut(out_ptr, FEAT_DIM);
    extract_recording_features(traces, shots, nt, nrec, input_scale, noise_frac, seed, out);
}

/// # Safety
/// All pointers must reference valid slices for `k_side`.
#[no_mangle]
pub unsafe extern "C" fn wasm_predict_gaussian_prior(
    feat_ptr: *const f32,
    base_ptr: *const f32,
    target_ptr: *const f32,
    k_side: usize,
    params_ptr: *const f32,
    pred_ptr: *mut f32,
    gauss_ptr: *mut f32,
    stats_ptr: *mut f32,
) {
    let feat = std::slice::from_raw_parts(feat_ptr, FEAT_DIM);
    let base = std::slice::from_raw_parts(base_ptr, CELLS);
    let target = std::slice::from_raw_parts(target_ptr, CELLS);
    let params = std::slice::from_raw_parts(params_ptr, prior_param_count(k_side));
    let pred = std::slice::from_raw_parts_mut(pred_ptr, CELLS);
    let gauss = std::slice::from_raw_parts_mut(gauss_ptr, k_side * k_side * SLOTS);
    let stats = std::slice::from_raw_parts_mut(stats_ptr, 3);
    let (loss, rmse_mps, edge_rmse_mps) =
        predict_gaussian_prior(feat, base, target, k_side, params, pred, gauss);
    stats[0] = loss;
    stats[1] = rmse_mps;
    stats[2] = edge_rmse_mps;
}

/// # Safety
/// All pointers must reference valid slices for `k_side`.
#[no_mangle]
pub unsafe extern "C" fn wasm_train_gaussian_prior_step(
    feat_ptr: *const f32,
    base_ptr: *const f32,
    target_ptr: *const f32,
    k_side: usize,
    step_idx: u32,
    lr: f32,
    weight_decay: f32,
    params_ptr: *mut f32,
    m1_ptr: *mut f32,
    m2_ptr: *mut f32,
    pred_ptr: *mut f32,
    gauss_ptr: *mut f32,
    stats_ptr: *mut f32,
) {
    let feat = std::slice::from_raw_parts(feat_ptr, FEAT_DIM);
    let base = std::slice::from_raw_parts(base_ptr, CELLS);
    let target = std::slice::from_raw_parts(target_ptr, CELLS);
    let n_params = prior_param_count(k_side);
    let params = std::slice::from_raw_parts_mut(params_ptr, n_params);
    let m1 = std::slice::from_raw_parts_mut(m1_ptr, n_params);
    let m2 = std::slice::from_raw_parts_mut(m2_ptr, n_params);
    let pred = std::slice::from_raw_parts_mut(pred_ptr, CELLS);
    let gauss = std::slice::from_raw_parts_mut(gauss_ptr, k_side * k_side * SLOTS);
    let stats = std::slice::from_raw_parts_mut(stats_ptr, 3);
    let (loss, rmse_mps, edge_rmse_mps) = train_gaussian_prior_step(
        feat,
        base,
        target,
        k_side,
        step_idx,
        lr,
        weight_decay,
        params,
        m1,
        m2,
        pred,
        gauss,
    );
    stats[0] = loss;
    stats[1] = rmse_mps;
    stats[2] = edge_rmse_mps;
}

/// Independent CPU reference for WebGPU renderer/gradient verification.
/// # Safety
/// Caller supplies CELLS floats for base/target/output; count*SLOTS for
/// gaussians/gradient; three floats for loss/RMSE/edge RMSE.
#[no_mangle]
pub unsafe extern "C" fn wasm_splat_reference(
    base: *const f32, target: *const f32, gaussians: *const f32, count: usize,
    prediction: *mut f32, gradient: *mut f32, statistics: *mut f32,
) {
    let (loss, rmse, edge) = splat_and_loss_grad(
        std::slice::from_raw_parts(base, CELLS),
        std::slice::from_raw_parts(target, CELLS),
        std::slice::from_raw_parts(gaussians, count * SLOTS),
        std::slice::from_raw_parts_mut(prediction, CELLS),
        std::slice::from_raw_parts_mut(gradient, count * SLOTS), true);
    std::slice::from_raw_parts_mut(statistics, 3).copy_from_slice(&[loss, rmse, edge]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_field_blur_and_zero_gradient() {
        let v = vec![2500.0f32; CELLS];
        let mut blur = vec![0.0f32; CELLS];
        let mut anom = vec![0.0f32; CELLS];
        let mut grad = vec![0.0f32; CELLS];
        let s = analyze_model(&v, 8.0, &mut blur, &mut anom, &mut grad);
        assert!((s.min_v - 2500.0).abs() < 1e-3);
        assert!((s.max_v - 2500.0).abs() < 1e-3);
        assert!((s.mean_v - 2500.0).abs() < 1e-3);
        assert!(s.rms_anomaly < 5e-3);
        assert!(s.rms_gradient < 1e-5);
        assert!(s.valid_bounds);
    }

    #[test]
    fn step_interface_has_nonzero_anomaly_and_gradient() {
        let mut v = vec![2000.0f32; CELLS];
        for z in 35..N {
            for x in 0..N {
                v[z * N + x] = 3500.0;
            }
        }
        let mut blur = vec![0.0f32; CELLS];
        let mut anom = vec![0.0f32; CELLS];
        let mut grad = vec![0.0f32; CELLS];
        let s = analyze_model(&v, 4.0, &mut blur, &mut anom, &mut grad);
        assert_eq!(s.min_v, 2000.0);
        assert_eq!(s.max_v, 3500.0);
        assert!((s.mean_v - 2750.0).abs() < 1e-3);
        assert!(s.rms_anomaly > 50.0);
        assert!(s.rms_gradient > 50.0);
        assert!((s.max_gradient - 750.0).abs() < 1e-3);
    }

    #[test]
    fn deterministic_hash_and_disjoint_splits() {
        let v1 = vec![2000.0f32; CELLS];
        let mut v2 = v1.clone();
        v2[100] = 2001.0;
        assert_eq!(fnv1a_64(&v1), fnv1a_64(&v1));
        assert_ne!(fnv1a_64(&v1), fnv1a_64(&v2));

        let idx = sample_unique_indices(500, 125, 42);
        assert_eq!(idx.len(), 125);
        assert_eq!(idx, sample_unique_indices(500, 125, 42));
        let mut sorted = idx.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 125);
    }

    #[test]
    fn ricker_cpml_and_trace_noise() {
        let mut w = vec![0.0f32; 2001];
        ricker_wavelet(2001, 0.0005, 15.0, &mut w);
        // Peak at t = 1.5 / 15 = 0.1 s -> index 200
        assert!((w[200] - 1.0).abs() < 1e-6);
        assert!(w[0].abs() < 1e-3);

        let n = 114;
        let (mut a, mut b, mut db) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        cpml_profiles(n, 10.0, 0.0005, 20, 5000.0, 15.0, &mut a, &mut b, &mut db);
        // Interior node 55 has zero damping
        assert_eq!(a[55], 0.0);
        assert_eq!(b[55], 0.0);
        assert!(a[5] > 0.0 && b[5] < 0.0);

        let clean: Vec<f32> = (0..10_000).map(|i| ((i as f32) * 0.01).sin()).collect();
        let mut noisy = vec![0.0f32; clean.len()];
        let stats = apply_trace_noise(&clean, 0.02, 42, &mut noisy);
        let ratio = stats.noise_rms / stats.clean_rms;
        assert!((ratio - 0.02).abs() < 0.001);
        assert!((stats.snr_db - 33.98).abs() < 0.5);
    }

    #[test]
    fn cpu_wave_solver_propagates_and_absorbs() {
        let v = vec![2500.0f32; CELLS];
        let nt = 400;
        let mut traces = vec![0.0f32; nt * N];
        simulate_shot_cpu(&v, 10.0, 0.0005, nt, 20, 1000.0, 5000.0, 15.0, 340.0, 10.0, &mut traces);
        let energy: f64 = traces.iter().map(|&x| (x as f64) * (x as f64)).sum();
        assert!(energy > 1e-4, "wavefield must reach surface receivers");
        assert!(traces.iter().all(|x| x.is_finite()), "traces must remain finite");
    }

    #[test]
    fn splat_and_loss_gradient_matches_finite_differences() {
        let base_v = vec![2500.0f32; CELLS];
        let mut target_v = vec![2500.0f32; CELLS];
        for z in 0..N {
            for x in 0..N {
                let dx = (x as f32) * H - 320.0;
                let dz = (z as f32) * H - 300.0;
                target_v[z * N + x] += 400.0 * (-0.5 * (dx * dx + dz * dz) / (70.0 * 70.0)).exp();
            }
        }
        let g = vec![280.0f32, 300.0, 310.0, (65.0f32).ln(), (80.0f32).ln(), 0.25, 0.0, 0.0];
        let mut pred = vec![0.0f32; CELLS];
        let mut grad = vec![0.0f32; SLOTS];
        splat_and_loss_grad(&base_v, &target_v, &g, &mut pred, &mut grad, true);

        let eps = [1.0f32, 0.2, 0.2, 1e-3, 1e-3, 1e-3];
        for p in 0..6 {
            let mut gp = g.clone();
            let mut gm = g.clone();
            gp[p] += eps[p];
            gm[p] -= eps[p];
            let (lp, _, _) = splat_and_loss_grad(&base_v, &target_v, &gp, &mut pred, &mut [], false);
            let (lm, _, _) = splat_and_loss_grad(&base_v, &target_v, &gm, &mut pred, &mut [], false);
            let fd = (lp - lm) / (2.0 * eps[p]);
            let rel = (grad[p] - fd).abs() / fd.abs().max(1e-6);
            assert!(rel < 2e-2, "slot {p}: analytical {} vs FD {} (rel {rel})", grad[p], fd);
        }
    }

    #[test]
    fn gaussian_prior_reduces_error_on_synthetic_target() {
        let base_v = vec![2500.0f32; CELLS];
        let mut target_v = vec![2500.0f32; CELLS];
        for z in 20..50 {
            for x in 15..55 {
                target_v[z * N + x] = 3200.0;
            }
        }
        let feat: Vec<f32> = (0..FEAT_DIM).map(|i| ((i as f32) * 0.13).sin() * 0.5).collect();
        let k_side = 6;
        let n_params = prior_param_count(k_side);
        let mut params = vec![0.0f32; n_params];
        let mut m1 = vec![0.0f32; n_params];
        let mut m2 = vec![0.0f32; n_params];
        let mut pred = vec![0.0f32; CELLS];
        let mut gauss = vec![0.0f32; k_side * k_side * SLOTS];
        init_gaussian_prior(k_side, 42, &mut params);

        let (_, rmse0, _) = predict_gaussian_prior(&feat, &base_v, &target_v, k_side, &params, &mut pred, &mut gauss);
        let mut rmse_last = rmse0;
        for step in 1..=25 {
            let (_, rmse, _) = train_gaussian_prior_step(
                &feat,
                &base_v,
                &target_v,
                k_side,
                step,
                0.005,
                1e-4,
                &mut params,
                &mut m1,
                &mut m2,
                &mut pred,
                &mut gauss,
            );
            rmse_last = rmse;
        }
        assert!(rmse_last < rmse0 * 0.65, "expected RMSE to drop significantly: {rmse0} -> {rmse_last}");
    }
}
