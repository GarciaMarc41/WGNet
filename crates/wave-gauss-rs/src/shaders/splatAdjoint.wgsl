// Adjoint time step of the splat inversion: adjoint.wgsl (the exact discrete
// adjoint of deepwave's propagator, validated against its autograd gradient),
// with splatForward.wgsl's interpolated stations: each receiver's residual
// enters its four corners with their weights (a per-cell list, so cells never
// race), and the source term of the gradient uses the source's corner weights.

struct Params {
  nx: u32,
  nz: u32,
  rh: f32,
  dt2: f32,
  nrec: u32,
  nt: u32,
  region: u32,
  shots: u32,
  store_every: u32,
  nsnap: u32,
  store_offset: u32,
  region_adj: u32,
  src: u32,
  rec: u32,
  csr: u32,
  entries: u32,
};

struct Step { t: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<uniform> step: Step;
@group(0) @binding(2) var<storage, read> statics: array<f32>;
@group(0) @binding(3) var<storage, read> geometry: array<u32>;
@group(0) @binding(4) var<storage, read> cur: array<f32>;        // lambda at step t
@group(0) @binding(5) var<storage, read_write> prev: array<f32>; // in: lambda_prev, out: next
@group(0) @binding(6) var<storage, read_write> psi: array<f32>;
@group(0) @binding(7) var<storage, read_write> zeta: array<f32>;
// Residuals dLoss/dtraces (shot, t, receiver), then the forward's stored terms.
@group(0) @binding(8) var<storage, read> outputs: array<f32>;
// Gradient (shot, cell), then the pseudo-Hessian (shot, cell).
@group(0) @binding(9) var<storage, read_write> grad: array<f32>;

const FD_PAD: u32 = 2u;
const D1A: f32 = 0.6666666666666666;
const D1B: f32 = -0.08333333333333333;
const D2A: f32 = -2.5;
const D2B: f32 = 1.3333333333333333;
const D2C: f32 = -0.08333333333333333;

fn profile(offset: u32) -> f32 { return statics[p.nx * p.nz + offset]; }
fn prof_a(axis: u32, k: u32) -> f32 { return select(profile(3u * p.nx + k), profile(k), axis == 0u); }
fn prof_b(axis: u32, k: u32) -> f32 { return select(profile(3u * p.nx + p.nz + k), profile(p.nx + k), axis == 0u); }
fn prof_db(axis: u32, k: u32) -> f32 { return select(profile(3u * p.nx + 2u * p.nz + k), profile(2u * p.nx + k), axis == 0u); }

fn vw(field: u32, i: u32) -> f32 {
  let v = statics[i];
  return v * v * p.dt2 * cur[field + i];
}

@compute @workgroup_size(16, 16, 1)
fn adjoint(@builtin(global_invocation_id) id: vec3<u32>) {
  let x = id.x;
  let z = id.y;
  let s = id.z;
  if (x >= p.nx || z >= p.nz || s >= p.shots) { return; }
  if (x < FD_PAD || z < FD_PAD || x >= p.nx - FD_PAD || z >= p.nz - FD_PAD) { return; }
  let cells = p.nx * p.nz;
  let field = s * cells;
  let i = z * p.nx + x;
  let c = field + i;
  let t = step.t;
  let q = (p.nt - 1u - t) & 1u;
  let half = p.shots * 2u * cells;
  let rd = q * half + s * 2u * cells;
  let wr = (1u - q) * half + s * 2u * cells;

  var w = 0.0;
  for (var axis = 0u; axis < 2u; axis++) {
    let stride = select(p.nx, 1u, axis == 0u);
    let along = select(z, x, axis == 0u);
    let n = select(p.nz, p.nx, axis == 0u);
    let v0 = vw(field, i);
    let vp1 = vw(field, i + stride);
    let vm1 = vw(field, i - stride);
    let vp2 = vw(field, i + 2u * stride);
    let vm2 = vw(field, i - 2u * stride);
    if (along < p.region_adj || along >= n - p.region_adj) {
      let base = rd + axis * cells + i;
      var tt: array<f32, 5>;
      var u1: array<f32, 5>;
      var u2: array<f32, 5>;
      var vs = array<f32, 5>(vm2, vm1, v0, vp1, vp2);
      for (var d = 0u; d < 5u; d++) {
        let k = along + d - 2u;
        let off = base + d * stride - 2u * stride;
        let b = prof_b(axis, k);
        let term = (1.0 + b) * vs[d] + b * zeta[off];
        tt[d] = term;
        u1[d] = prof_db(axis, k) * term + b * psi[off];
        u2[d] = (1.0 + b) * term;
      }
      let d1_u1 = (D1A * (u1[3] - u1[1]) + D1B * (u1[4] - u1[0])) * p.rh;
      let d2_u2 = (D2A * u2[2] + D2B * (u2[3] + u2[1]) + D2C * (u2[4] + u2[0])) * p.rh * p.rh;
      let d1_t = (D1A * (tt[3] - tt[1]) + D1B * (tt[4] - tt[0])) * p.rh;
      w += -d1_u1 + d2_u2;
      let a = prof_a(axis, along);
      psi[wr + axis * cells + i] = -a * d1_t + a * psi[base];
      zeta[wr + axis * cells + i] = a * v0 + a * zeta[base];
    } else {
      w += (D2A * v0 + D2B * (vp1 + vm1) + D2C * (vp2 + vm2)) * p.rh * p.rh;
    }
  }
  var next = w + 2.0 * cur[c] - prev[c];
  var g = 0.0;
  if (p.store_every > 0u && t % p.store_every == 0u) {
    let stored = outputs[p.store_offset + (s * p.nsnap + t / p.store_every) * cells + i];
    g += cur[c] * stored * f32(p.store_every);
    grad[p.shots * cells + c] += stored * stored * f32(p.store_every);
  }
  for (var e = geometry[p.csr + i]; e < geometry[p.csr + i + 1u]; e++) {
    let receiver = geometry[p.entries + 2u * e];
    next += bitcast<f32>(geometry[p.entries + 2u * e + 1u]) * outputs[(s * p.nt + t) * p.nrec + receiver];
  }
  let v = statics[i];
  let f = statics[cells + 3u * (p.nx + p.nz) + t];
  for (var k = 0u; k < 4u; k++) {
    if (i == geometry[p.src + 4u * s + k]) {
      g += cur[c] * (-2.0 * v * p.dt2 * bitcast<f32>(geometry[p.src + 4u * p.shots + 4u * s + k]) * f);
    }
  }
  grad[c] += g;
  prev[c] = next;
}
