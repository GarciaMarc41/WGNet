//! Batched serpentine Floyd-Steinberg dithering, Delaunay triangulation,
//! closed-form Steiner circumellipse priors, and K=10 nearest-neighbor
//! teacher density estimation, parallelized across batch items via Rayon.

use delaunator::{triangulate, Point};
use rayon::prelude::*;

pub const ROOT_SIZE: usize = 8;
pub const TEACHER_RMSE_KM: f64 = 0.025; // 25 m/s in km/s
pub const INITIAL_WIDTH_FRACTION: f32 = 0.15;

#[derive(Clone, Debug)]
pub struct DelaunayLayout {
    pub height: usize,
    pub width: usize,
    pub count: usize,
    pub sampled_point_count: usize,
    pub density_sum: f32,
    pub selected_pixels: Vec<u32>,
    pub triangles: Vec<u32>,
    pub vertices: Vec<f32>,           // [count * 6]
    pub priors: Vec<f32>,             // [count * 5]: [cx, cz, angle, log_sx, log_sz]
    pub sample_coordinates: Vec<f32>, // [count * 8]: [x0, z0, x1, z1, x2, z2, cx, cz]
}

#[derive(Clone, Debug)]
pub struct BatchedDelaunayLayout {
    pub batch_size: usize,
    pub padded_count: usize,
    pub counts: Vec<u32>,            // [B]
    pub vertices: Vec<f32>,          // [B * N_pad * 6]
    pub priors: Vec<f32>,            // [B * N_pad * 5]
    pub sample_coords: Vec<f32>,     // [B * N_pad * 8]
    pub mask: Vec<f32>,              // [B * N_pad] (0.0 for valid, -1e30 for padded)
}

/// Serpentine Floyd-Steinberg error-diffusion dithering with valid-neighbor weight renormalization.
/// Matches `ditherDensity` in `src/direct/geometry.ts`.
pub fn dither_density(ppm: &[f32], height: usize, width: usize) -> Vec<u32> {
    assert_eq!(ppm.len(), height * width, "Density length must match height * width");
    let mut values: Vec<f64> = ppm
        .iter()
        .map(|&x| (x as f64).clamp(0.0, 1.0))
        .collect();
    let mut selected = Vec::new();

    for z in 0..height {
        let direction: isize = if z % 2 == 1 { -1 } else { 1 };
        for s in 0..width {
            let x: isize = if direction == 1 {
                s as isize
            } else {
                (width - 1 - s) as isize
            };
            let id = (z as isize * width as isize + x) as usize;
            let bit = if values[id] >= 0.5 { 1.0 } else { 0.0 };
            let error = values[id] - bit;
            if bit > 0.5 {
                selected.push(id as u32);
            }

            let neighbors: [(isize, isize, f64); 4] = [
                (x + direction, z as isize, 7.0),
                (x - direction, z as isize + 1, 3.0),
                (x, z as isize + 1, 5.0),
                (x + direction, z as isize + 1, 1.0),
            ];
            let mut weight = 0.0f64;
            for &(nx, nz, w) in &neighbors {
                if nx >= 0 && nx < width as isize && nz >= 0 && nz < height as isize {
                    weight += w;
                }
            }
            if weight > 0.0 {
                for &(nx, nz, w) in &neighbors {
                    if nx >= 0 && nx < width as isize && nz >= 0 && nz < height as isize {
                        let nid = (nz as usize) * width + (nx as usize);
                        values[nid] += error * w / weight;
                    }
                }
            }
        }
    }

    selected.sort_unstable();
    selected
}

/// Closed-form Steiner circumellipse prior for a triangle `[x0, z0, x1, z1, x2, z2]`.
/// Returns `[cx, cz, angle, log_sigma_major, log_sigma_minor]`.
#[inline]
pub fn triangle_prior(vertices: &[f32; 6]) -> [f32; 5] {
    let cx = ((vertices[0] as f64) + (vertices[2] as f64) + (vertices[4] as f64)) / 3.0;
    let cz = ((vertices[1] as f64) + (vertices[3] as f64) + (vertices[5] as f64)) / 3.0;
    let mut xx = 0.0f64;
    let mut xz = 0.0f64;
    let mut zz = 0.0f64;
    for i in 0..3 {
        let dx = (vertices[2 * i] as f64) - cx;
        let dz = (vertices[2 * i + 1] as f64) - cz;
        xx += dx * dx * (2.0 / 3.0);
        xz += dx * dz * (2.0 / 3.0);
        zz += dz * dz * (2.0 / 3.0);
    }
    let delta = (xx - zz).hypot(2.0 * xz);
    let major = (xx + zz + delta) * 0.5;
    let minor = ((xx * zz - xz * xz) / major.max(f64::MIN_POSITIVE)).max(0.0);
    let angle = if delta <= 1e-14 * (xx + zz) {
        0.0
    } else {
        0.5 * (2.0 * xz).atan2(xx - zz)
    };
    let log_major = (0.5 * major.sqrt()).max(0.08).ln() as f32;
    let log_minor = (0.5 * minor.sqrt()).max(0.08).ln() as f32;
    [cx as f32, cz as f32, angle as f32, log_major, log_minor]
}

/// Discretize a single `[height, width]` probability map and triangulate it with 4 corner anchors.
pub fn density_to_layout(ppm: &[f32], height: usize, width: usize) -> DelaunayLayout {
    let selected_pixels = dither_density(ppm, height, width);
    let n_pts = selected_pixels.len() + 4;
    let mut points = Vec::with_capacity(n_pts);
    let w_f = width as f64;
    let h_f = height as f64;
    points.push(Point { x: 0.0, y: 0.0 });
    points.push(Point { x: w_f, y: 0.0 });
    points.push(Point { x: w_f, y: h_f });
    points.push(Point { x: 0.0, y: h_f });

    for &id in &selected_pixels {
        let px = (id as usize % width) as f64 + 0.5;
        let pz = (id as usize / width) as f64 + 0.5;
        points.push(Point { x: px, y: pz });
    }

    let result = triangulate(&points);
    let num_tris = result.triangles.len() / 3;
    let mut ordered: Vec<[u32; 3]> = Vec::with_capacity(num_tris);

    for i in (0..result.triangles.len()).step_by(3) {
        let mut a = result.triangles[i] as u32;
        let mut b = result.triangles[i + 1] as u32;
        let mut c = result.triangles[i + 2] as u32;
        let pa = &points[a as usize];
        let pb = &points[b as usize];
        let pc = &points[c as usize];
        let cross = (pb.x - pa.x) * (pc.y - pa.y) - (pb.y - pa.y) * (pc.x - pa.x);
        if cross < 0.0 {
            std::mem::swap(&mut b, &mut c);
        }
        if b < a && b < c {
            let tmp = a;
            a = b;
            b = c;
            c = tmp;
        } else if c < a && c < b {
            let tmp = a;
            a = c;
            c = b;
            b = tmp;
        }
        ordered.push([a, b, c]);
    }

    ordered.sort_unstable();
    let count = ordered.len();
    debug_assert_eq!(count, 2 * selected_pixels.len() + 2);

    let mut triangles = Vec::with_capacity(count * 3);
    let mut vertices = Vec::with_capacity(count * 6);
    let mut priors = Vec::with_capacity(count * 5);
    let mut sample_coordinates = Vec::with_capacity(count * 8);

    for tri in &ordered {
        triangles.extend_from_slice(tri);
        let mut v6 = [0.0f32; 6];
        for (j, &idx) in tri.iter().enumerate() {
            let pt = &points[idx as usize];
            v6[2 * j] = pt.x as f32;
            v6[2 * j + 1] = pt.y as f32;
        }
        let prior = triangle_prior(&v6);
        vertices.extend_from_slice(&v6);
        priors.extend_from_slice(&prior);
        sample_coordinates.extend_from_slice(&v6);
        sample_coordinates.push(prior[0]);
        sample_coordinates.push(prior[1]);
    }

    let density_sum: f32 = ppm.iter().copied().sum();
    DelaunayLayout {
        height,
        width,
        count,
        sampled_point_count: selected_pixels.len(),
        density_sum,
        selected_pixels,
        triangles,
        vertices,
        priors,
        sample_coordinates,
    }
}

/// Batched parallel dithering + Delaunay triangulation + Steiner priors across `batch_size` maps.
pub fn batch_density_to_layout(
    batch_ppm: &[f32],
    batch_size: usize,
    height: usize,
    width: usize,
    bucket_multiple: usize,
) -> BatchedDelaunayLayout {
    let hw = height * width;
    assert_eq!(batch_ppm.len(), batch_size * hw);
    let bucket = bucket_multiple.max(1);

    let layouts: Vec<DelaunayLayout> = (0..batch_size)
        .into_par_iter()
        .map(|b| density_to_layout(&batch_ppm[b * hw..(b + 1) * hw], height, width))
        .collect();

    let max_count = layouts.iter().map(|l| l.count).max().unwrap_or(2);
    let padded_count = max_count.div_ceil(bucket) * bucket;

    let mut counts = Vec::with_capacity(batch_size);
    let mut vertices = vec![0.0f32; batch_size * padded_count * 6];
    let mut priors = vec![0.0f32; batch_size * padded_count * 5];
    let mut sample_coords = vec![0.0f32; batch_size * padded_count * 8];
    let mut mask = vec![-1.0e30f32; batch_size * padded_count];

    for (b, layout) in layouts.iter().enumerate() {
        let c = layout.count;
        counts.push(c as u32);
        let v_off = b * padded_count * 6;
        let p_off = b * padded_count * 5;
        let s_off = b * padded_count * 8;
        let m_off = b * padded_count;

        vertices[v_off..v_off + c * 6].copy_from_slice(&layout.vertices);
        priors[p_off..p_off + c * 5].copy_from_slice(&layout.priors);
        sample_coords[s_off..s_off + c * 8].copy_from_slice(&layout.sample_coordinates);
        for k in 0..c {
            mask[m_off + k] = 0.0;
        }
    }

    BatchedDelaunayLayout {
        batch_size,
        padded_count,
        counts,
        vertices,
        priors,
        sample_coords,
        mask,
    }
}

/// Build automatic quadtree teacher Gaussians `[x, z, angle, log_sx, log_sz, mean]` for an observed velocity field in km/s.
/// Matches `automaticParams(makeTree(fieldKm, [height, width]))` in `src/spatial/layout.ts`.
pub fn quadtree_teacher_params(field_km: &[f32], height: usize, width: usize) -> Vec<f32> {
    assert_eq!(field_km.len(), height * width);
    let mut params = Vec::new();

    fn visit_cell(
        field_km: &[f32],
        height: usize,
        width: usize,
        left: usize,
        top: usize,
        size: usize,
        out: &mut Vec<f32>,
    ) {
        let w = size.min(width - left);
        let h = size.min(height - top);
        let count = (w * h) as f64;
        let mut sum = 0.0f64;
        for z in top..(top + h) {
            for x in left..(left + w) {
                sum += field_km[z * width + x] as f64;
            }
        }
        let mean = sum / count;
        let mut var = 0.0f64;
        for z in top..(top + h) {
            for x in left..(left + w) {
                let d = (field_km[z * width + x] as f64) - mean;
                var += d * d;
            }
        }
        var /= count;

        if size > 1 && var > TEACHER_RMSE_KM * TEACHER_RMSE_KM {
            let half = size / 2;
            for dz in [0, half] {
                for dx in [0, half] {
                    if left + dx < width && top + dz < height {
                        visit_cell(field_km, height, width, left + dx, top + dz, half, out);
                    }
                }
            }
        } else {
            let cx = (left as f32) + (w as f32) * 0.5;
            let cz = (top as f32) + (h as f32) * 0.5;
            let log_sx = (INITIAL_WIDTH_FRACTION * (w as f32)).ln();
            let log_sz = (INITIAL_WIDTH_FRACTION * (h as f32)).ln();
            out.extend_from_slice(&[cx, cz, 0.0, log_sx, log_sz, mean as f32]);
        }
    }

    for top in (0..height).step_by(ROOT_SIZE) {
        for left in (0..width).step_by(ROOT_SIZE) {
            visit_cell(field_km, height, width, left, top, ROOT_SIZE, &mut params);
        }
    }
    params
}

/// Convert teacher Gaussian parameters `[N * 6]` into K=10 nearest-neighbor count-calibrated density map `[height * width]`.
/// Matches `densityFromTeacher` in `src/direct/geometry.ts`.
pub fn density_from_teacher(
    params: &[f32],
    height: usize,
    width: usize,
    neighbors: usize,
) -> Vec<f32> {
    assert!(params.len() >= 6 && params.len() % 6 == 0);
    let count = params.len() / 6;
    let k = neighbors.max(1).min(count);
    let hw = height * width;

    let centers: Vec<(f64, f64)> = (0..count)
        .map(|i| (params[i * 6] as f64, params[i * 6 + 1] as f64))
        .collect();

    let mut raw_density = vec![0.0f64; hw];
    let mut dists = vec![f64::INFINITY; k];

    for z in 0..height {
        let qz = (z as f64) + 0.5;
        for x in 0..width {
            let qx = (x as f64) + 0.5;
            dists.fill(f64::INFINITY);
            for &(cx, cz) in &centers {
                let dx = qx - cx;
                let dz = qz - cz;
                let d2 = dx * dx + dz * dz;
                if d2 < dists[k - 1] {
                    let mut at = k - 1;
                    while at > 0 && d2 < dists[at - 1] {
                        dists[at] = dists[at - 1];
                        at -= 1;
                    }
                    dists[at] = d2;
                }
            }
            let r2 = dists[k - 1].max(0.25);
            raw_density[z * width + x] = (k as f64) / (std::f64::consts::PI * r2);
        }
    }

    let desired = (((count as f64) - 2.0) * 0.5).clamp(0.0, hw as f64);
    if desired <= 0.0 {
        return vec![0.0f32; hw];
    }

    let mass = |scale: f64| -> f64 {
        raw_density
            .iter()
            .map(|&v| (scale * v).min(1.0))
            .sum::<f64>()
    };

    let mut low = 0.0f64;
    let mut high = 1.0f64;
    while mass(high) < desired && high < 1e30 {
        high *= 2.0;
    }
    for _ in 0..50 {
        let mid = 0.5 * (low + high);
        if mass(mid) < desired {
            low = mid;
        } else {
            high = mid;
        }
    }
    let scale = 0.5 * (low + high);
    raw_density
        .into_iter()
        .map(|v| (scale * v).min(1.0) as f32)
        .collect()
}

/// Compute teacher density maps for a batch of velocity fields in km/s `[B * height * width]` in parallel via Rayon.
pub fn batch_teacher_densities(
    fields_km: &[f32],
    batch_size: usize,
    height: usize,
    width: usize,
    neighbors: usize,
) -> Vec<f32> {
    let hw = height * width;
    assert_eq!(fields_km.len(), batch_size * hw);
    let rows: Vec<Vec<f32>> = (0..batch_size)
        .into_par_iter()
        .map(|b| {
            let field = &fields_km[b * hw..(b + 1) * hw];
            let params = quadtree_teacher_params(field, height, width);
            density_from_teacher(&params, height, width, neighbors)
        })
        .collect();
    rows.into_iter().flatten().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dithering_and_triangulation_conservation() {
        let ppm = vec![0.15f32; 70 * 70];
        let layout = density_to_layout(&ppm, 70, 70);
        assert_eq!(layout.count, 2 * layout.sampled_point_count + 2);
        assert_eq!(layout.vertices.len(), layout.count * 6);
        assert_eq!(layout.priors.len(), layout.count * 5);
        assert!(layout.priors.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_teacher_density_calibration() {
        let mut field = vec![2.5f32; 70 * 70];
        for z in 35..70 {
            for x in 0..70 {
                field[z * 70 + x] = 3.5;
            }
        }
        let params = quadtree_teacher_params(&field, 70, 70);
        let count = params.len() / 6;
        let density = density_from_teacher(&params, 70, 70, 10);
        let sum: f32 = density.iter().sum();
        let expected = ((count as f32) - 2.0) * 0.5;
        assert!((sum - expected).abs() < 1e-2, "sum={} expected={}", sum, expected);
    }
}
