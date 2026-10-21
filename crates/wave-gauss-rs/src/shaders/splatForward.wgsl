// Forward time step of the splat inversion: wave.wgsl's scheme (deepwave's
// scalar propagator with its C-PML), unchanged, except that sources and
// receivers sit between grid nodes. Each has four bilinear corners: a source
// adds its wavelet to them with their weights, a receiver records the
// weighted sum of u there. One dispatch per step, every shot at once (z index).

struct Params {
  nx: u32,
  nz: u32,
  rh: f32,          // 1 / h
  dt2: f32,         // dt^2
  nrec: u32,
  nt: u32,
  region: u32,      // forward PML region (width + 2 FD_PAD)
  shots: u32,
  store_every: u32,
  nsnap: u32,
  store_offset: u32,
  region_adj: u32,  // backward PML region (width + 3 FD_PAD)
  src: u32,         // geometry: 4 corner cells per shot, then their 4 weights
  rec: u32,         // 4 corner cells per receiver, then their 4 weights
  csr: u32,         // per cell, first receiver entry (cells + 1 words)
  entries: u32,     // (receiver, weight) pairs, ordered by cell
};

struct Step { t: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<uniform> step: Step;
// velocity (nx * nz) | C-PML profiles ax bx dbx (nx each), az bz dbz (nz each) | wavelet (nt)
@group(0) @binding(2) var<storage, read> statics: array<f32>;
@group(0) @binding(3) var<storage, read> geometry: array<u32>;
@group(0) @binding(4) var<storage, read> cur: array<f32>;
@group(0) @binding(5) var<storage, read_write> prev: array<f32>;   // in: u_prev, out: u_next
@group(0) @binding(6) var<storage, read> psi: array<f32>;
@group(0) @binding(7) var<storage, read_write> psi_next: array<f32>;
@group(0) @binding(8) var<storage, read_write> zeta: array<f32>;
// Traces (shot, t, receiver), then the stored gradient terms (shot, snapshot, cell).
@group(0) @binding(9) var<storage, read_write> outputs: array<f32>;

const FD_PAD: u32 = 2u;

fn profile(offset: u32) -> f32 { return statics[p.nx * p.nz + offset]; }
fn prof_a(axis: u32, k: u32) -> f32 { return select(profile(3u * p.nx + k), profile(k), axis == 0u); }
fn prof_b(axis: u32, k: u32) -> f32 { return select(profile(3u * p.nx + p.nz + k), profile(p.nx + k), axis == 0u); }
fn prof_db(axis: u32, k: u32) -> f32 { return select(profile(3u * p.nx + 2u * p.nz + k), profile(2u * p.nx + k), axis == 0u); }

@compute @workgroup_size(16, 16, 1)
fn update(@builtin(global_invocation_id) id: vec3<u32>) {
  let x = id.x;
  let z = id.y;
  let s = id.z;
  if (x >= p.nx || z >= p.nz || s >= p.shots) { return; }
  let cells = p.nx * p.nz;
  let field = s * cells;
  let aux = s * 2u * cells;
  let i = z * p.nx + x;
  let t = step.t;

  // The first nrec invocations of this shot record the receivers.
  if (i < p.nrec && t < p.nt) {
    var r = 0.0;
    for (var k = 0u; k < 4u; k++) {
      r += bitcast<f32>(geometry[p.rec + 4u * p.nrec + 4u * i + k]) * cur[field + geometry[p.rec + 4u * i + k]];
    }
    outputs[(s * p.nt + t) * p.nrec + i] = r;
  }

  if (x < FD_PAD || z < FD_PAD || x >= p.nx - FD_PAD || z >= p.nz - FD_PAD) {
    prev[field + i] = 0.0;
    psi_next[aux + i] = 0.0;
    psi_next[aux + cells + i] = 0.0;
    return;
  }
  let c = field + i;
  let u = cur[c];
  var w = 0.0;
  for (var axis = 0u; axis < 2u; axis++) {
    let stride = select(p.nx, 1u, axis == 0u);
    let along = select(z, x, axis == 0u);
    let n = select(p.nz, p.nx, axis == 0u);
    let um1 = cur[c - stride];
    let up1 = cur[c + stride];
    let um2 = cur[c - 2u * stride];
    let up2 = cur[c + 2u * stride];
    let d2 = (-2.5 * u + 1.3333333333333333 * (up1 + um1) - 0.08333333333333333 * (up2 + um2)) * p.rh * p.rh;
    let a_i = aux + axis * cells + i;
    if (along < p.region || along >= n - p.region) {
      let d1 = (0.6666666666666666 * (up1 - um1) - 0.08333333333333333 * (up2 - um2)) * p.rh;
      let a = prof_a(axis, along);
      let b = prof_b(axis, along);
      let ap1 = prof_a(axis, along + 1u) * psi[a_i + stride];
      let am1 = prof_a(axis, along - 1u) * psi[a_i - stride];
      let ap2 = prof_a(axis, along + 2u) * psi[a_i + 2u * stride];
      let am2 = prof_a(axis, along - 2u) * psi[a_i - 2u * stride];
      let d1_apsi = (0.6666666666666666 * (ap1 - am1) - 0.08333333333333333 * (ap2 - am2)) * p.rh;
      let tmp = (1.0 + b) * d2 + prof_db(axis, along) * d1 + d1_apsi;
      w += (1.0 + b) * tmp + a * zeta[a_i];
      psi_next[a_i] = b * d1 + a * psi[a_i];
      zeta[a_i] = b * tmp + a * zeta[a_i];
    } else {
      w += d2;
      psi_next[a_i] = 0.0;
    }
  }
  let v = statics[i];
  let v2dt2 = v * v * p.dt2;
  if (p.store_every > 0u && t % p.store_every == 0u) {
    outputs[p.store_offset + (s * p.nsnap + t / p.store_every) * cells + i] = 2.0 * v * p.dt2 * w;
  }
  var next = v2dt2 * w + 2.0 * u - prev[c];
  if (t < p.nt) {
    let f = statics[cells + 3u * (p.nx + p.nz) + t];
    for (var k = 0u; k < 4u; k++) {
      if (i == geometry[p.src + 4u * s + k]) {
        next += -v2dt2 * bitcast<f32>(geometry[p.src + 4u * p.shots + 4u * s + k]) * f;
      }
    }
  }
  prev[c] = next;
}
