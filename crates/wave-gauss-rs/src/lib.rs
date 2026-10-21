//! `wave-gauss-rs`: PyO3 bindings for native `wgpu` multi-shot forward/adjoint wave solver,
//! Rayon-parallelized Floyd-Steinberg dithering + Delaunay triangulation + Steiner priors,
//! and OpenFWI split hashing / sampling from `geology-core`.

pub mod delaunay;
pub mod wave_wgpu;

use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
fn hash_model_hex(model: PyReadonlyArray1<'_, f32>) -> PyResult<String> {
    let slice = model
        .as_slice()
        .map_err(|_| PyValueError::new_err("model must be a contiguous 1D float32 array"))?;
    if slice.len() != geology_core::CELLS {
        return Err(PyValueError::new_err("model must have 4900 elements"));
    }
    let h = geology_core::fnv1a_64(slice);
    Ok(format!("{:016x}", h))
}

#[pyfunction]
fn sample_unique_indices(total: u32, pick: u32, seed: u32) -> Vec<u32> {
    geology_core::sample_unique_indices(total, pick, seed)
}

#[pyfunction]
fn ricker_wavelet<'py>(
    py: Python<'py>,
    samples: usize,
    dt: f32,
    peak_hz: f32,
) -> Bound<'py, PyArray1<f32>> {
    let mut out = vec![0.0f32; samples];
    geology_core::ricker_wavelet(samples, dt, peak_hz, &mut out);
    out.into_pyarray(py)
}

#[pyfunction]
fn simulate_shot_cpu<'py>(
    py: Python<'py>,
    vel_mps: PyReadonlyArray1<'_, f32>,
    h: f32,
    dt: f32,
    nt: usize,
    pml: usize,
    vmin: f32,
    vmax: f32,
    peak_hz: f32,
    src_x: f32,
    src_z: f32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let slice = vel_mps
        .as_slice()
        .map_err(|_| PyValueError::new_err("vel_mps must be contiguous"))?;
    if slice.len() != geology_core::CELLS {
        return Err(PyValueError::new_err("vel_mps must have 4900 elements"));
    }
    let mut out = vec![0.0f32; nt * geology_core::N];
    py.allow_threads(|| {
        geology_core::simulate_shot_cpu(
            slice, h, dt, nt, pml, vmin, vmax, peak_hz, src_x, src_z, &mut out,
        );
    });
    let arr = PyArray1::from_vec(py, out);
    arr.reshape([nt, geology_core::N])
}

#[pyfunction]
#[pyo3(signature = (ppm, height=70, width=70))]
fn dither_density<'py>(
    py: Python<'py>,
    ppm: PyReadonlyArray1<'_, f32>,
    height: usize,
    width: usize,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let slice = ppm
        .as_slice()
        .map_err(|_| PyValueError::new_err("ppm must be contiguous"))?;
    if slice.len() != height * width {
        return Err(PyValueError::new_err("ppm length does not match height * width"));
    }
    let selected = delaunay::dither_density(slice, height, width);
    Ok(selected.into_pyarray(py))
}

/// Discretize and triangulate a single `[height * width]` probability map.
/// Returns `(vertices [N, 3, 2], priors [N, 5], sample_coords [N, 4, 2], selected_pixels [M], count)`.
#[pyfunction]
#[pyo3(signature = (ppm, height=70, width=70))]
fn density_to_layout<'py>(
    py: Python<'py>,
    ppm: PyReadonlyArray1<'_, f32>,
    height: usize,
    width: usize,
) -> PyResult<(
    Bound<'py, PyArray3<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray3<f32>>,
    Bound<'py, PyArray1<u32>>,
    usize,
)> {
    let slice = ppm
        .as_slice()
        .map_err(|_| PyValueError::new_err("ppm must be contiguous"))?;
    if slice.len() != height * width {
        return Err(PyValueError::new_err("ppm length does not match height * width"));
    }
    let layout = py.allow_threads(|| delaunay::density_to_layout(slice, height, width));
    let c = layout.count;
    let vertices = PyArray1::from_vec(py, layout.vertices).reshape([c, 3, 2])?;
    let priors = PyArray1::from_vec(py, layout.priors).reshape([c, 5])?;
    let sample_coords = PyArray1::from_vec(py, layout.sample_coordinates).reshape([c, 4, 2])?;
    let selected = PyArray1::from_vec(py, layout.selected_pixels);
    Ok((vertices, priors, sample_coords, selected, c))
}

/// Parallel batched dithering + Delaunay + Steiner priors across `B` probability maps `[B, height * width]`.
/// Returns `(vertices [B, N_pad, 3, 2], priors [B, N_pad, 5], sample_coords [B, N_pad, 4, 2], mask [B, N_pad], counts [B])`.
#[pyfunction]
#[pyo3(signature = (batch_ppm, height=70, width=70, bucket_multiple=64))]
fn batch_density_to_layout<'py>(
    py: Python<'py>,
    batch_ppm: PyReadonlyArray2<'_, f32>,
    height: usize,
    width: usize,
    bucket_multiple: usize,
) -> PyResult<(
    Bound<'py, PyArray3<f32>>,
    Bound<'py, PyArray3<f32>>,
    Bound<'py, PyArray3<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<u32>>,
)> {
    let shape = batch_ppm.shape();
    let b = shape[0];
    if shape[1] != height * width {
        return Err(PyValueError::new_err(
            "batch_ppm second dimension must equal height * width",
        ));
    }
    let slice = batch_ppm
        .as_slice()
        .map_err(|_| PyValueError::new_err("batch_ppm must be contiguous"))?;
    let out = py.allow_threads(|| {
        delaunay::batch_density_to_layout(slice, b, height, width, bucket_multiple)
    });
    let npad = out.padded_count;
    let vertices = PyArray1::from_vec(py, out.vertices).reshape([b, npad, 6])?;
    let priors = PyArray1::from_vec(py, out.priors).reshape([b, npad, 5])?;
    let sample_coords = PyArray1::from_vec(py, out.sample_coords).reshape([b, npad, 8])?;
    let mask = PyArray1::from_vec(py, out.mask).reshape([b, npad])?;
    let counts = PyArray1::from_vec(py, out.counts);
    Ok((vertices, priors, sample_coords, mask, counts))
}

#[pyfunction]
#[pyo3(signature = (field_km, height=70, width=70))]
fn quadtree_teacher_params<'py>(
    py: Python<'py>,
    field_km: PyReadonlyArray1<'_, f32>,
    height: usize,
    width: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let slice = field_km
        .as_slice()
        .map_err(|_| PyValueError::new_err("field_km must be contiguous"))?;
    if slice.len() != height * width {
        return Err(PyValueError::new_err("field_km length mismatch"));
    }
    let params = delaunay::quadtree_teacher_params(slice, height, width);
    let count = params.len() / 6;
    PyArray1::from_vec(py, params).reshape([count, 6])
}

#[pyfunction]
#[pyo3(signature = (fields_km, height=70, width=70, neighbors=10))]
fn batch_teacher_densities<'py>(
    py: Python<'py>,
    fields_km: PyReadonlyArray2<'_, f32>,
    height: usize,
    width: usize,
    neighbors: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let shape = fields_km.shape();
    let b = shape[0];
    if shape[1] != height * width {
        return Err(PyValueError::new_err(
            "fields_km second dimension must equal height * width",
        ));
    }
    let slice = fields_km
        .as_slice()
        .map_err(|_| PyValueError::new_err("fields_km must be contiguous"))?;
    let densities = py.allow_threads(|| {
        delaunay::batch_teacher_densities(slice, b, height, width, neighbors)
    });
    PyArray1::from_vec(py, densities).reshape([b, height * width])
}

#[pyclass]
pub struct PyWaveEngine {
    inner: wave_wgpu::NativeWaveEngine,
}

#[pymethods]
impl PyWaveEngine {
    #[new]
    #[pyo3(signature = (shots=5, nt=2001, dt=0.0005, peak_hz=15.0, pml=20, vmin=1000.0, vmax=5000.0))]
    fn new(
        shots: usize,
        nt: usize,
        dt: f32,
        peak_hz: f32,
        pml: usize,
        vmin: f32,
        vmax: f32,
    ) -> PyResult<Self> {
        let inner = wave_wgpu::NativeWaveEngine::new(shots, nt, dt, peak_hz, pml, vmin, vmax)
            .map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }

    /// Update source positions `[shots, 2]` in meters.
    fn set_sources(&mut self, sources: PyReadonlyArray2<'_, f32>) -> PyResult<()> {
        let shape = sources.shape();
        if shape[0] != self.inner.shots || shape[1] != 2 {
            return Err(PyValueError::new_err("sources must have shape [shots, 2]"));
        }
        let slice = sources
            .as_slice()
            .map_err(|_| PyValueError::new_err("sources must be contiguous"))?;
        let pts: Vec<[f32; 2]> = (0..self.inner.shots)
            .map(|s| [slice[2 * s], slice[2 * s + 1]])
            .collect();
        self.inner.set_sources(&pts);
        Ok(())
    }

    /// Simulate all shots for a 70x70 velocity model in m/s.
    /// Returns `[shots, nt, nrec]` float32 array.
    fn simulate<'py>(
        &self,
        py: Python<'py>,
        vel_mps: PyReadonlyArray1<'_, f32>,
    ) -> PyResult<Bound<'py, PyArray3<f32>>> {
        let slice = vel_mps
            .as_slice()
            .map_err(|_| PyValueError::new_err("vel_mps must be contiguous"))?;
        if slice.len() != geology_core::CELLS {
            return Err(PyValueError::new_err("vel_mps must have 4900 elements"));
        }
        let traces = py.allow_threads(|| self.inner.simulate(slice));
        PyArray1::from_vec(py, traces).reshape([self.inner.shots, self.inner.nt, self.inner.nrec])
    }

    /// Run exact forward + discrete adjoint pass across all shots on GPU.
    /// Returns `(loss, grad_mps [70, 70])`.
    fn evaluate_adjoint<'py>(
        &self,
        py: Python<'py>,
        vel_mps: PyReadonlyArray1<'_, f32>,
        observed: PyReadonlyArray1<'_, f32>,
    ) -> PyResult<(f32, Bound<'py, PyArray2<f32>>)> {
        let v_slice = vel_mps
            .as_slice()
            .map_err(|_| PyValueError::new_err("vel_mps must be contiguous"))?;
        let obs_slice = observed
            .as_slice()
            .map_err(|_| PyValueError::new_err("observed must be contiguous"))?;
        if v_slice.len() != geology_core::CELLS {
            return Err(PyValueError::new_err("vel_mps must have 4900 elements"));
        }
        if obs_slice.len() != self.inner.shots * self.inner.nt * self.inner.nrec {
            return Err(PyValueError::new_err(
                "observed length does not match shots * nt * nrec",
            ));
        }
        let res = py.allow_threads(|| self.inner.evaluate_adjoint(v_slice, obs_slice));
        let grad = PyArray1::from_vec(py, res.grad_mps).reshape([70, 70])?;
        Ok((res.loss, grad))
    }
}

#[pymodule]
fn wave_gauss_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hash_model_hex, m)?)?;
    m.add_function(wrap_pyfunction!(sample_unique_indices, m)?)?;
    m.add_function(wrap_pyfunction!(ricker_wavelet, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_shot_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(dither_density, m)?)?;
    m.add_function(wrap_pyfunction!(density_to_layout, m)?)?;
    m.add_function(wrap_pyfunction!(batch_density_to_layout, m)?)?;
    m.add_function(wrap_pyfunction!(quadtree_teacher_params, m)?)?;
    m.add_function(wrap_pyfunction!(batch_teacher_densities, m)?)?;
    m.add_class::<PyWaveEngine>()?;
    Ok(())
}
