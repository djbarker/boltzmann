// Std imports:
use std::sync::{Arc, Mutex, MutexGuard};

use fields::{MemUsage, Scalar};
// Imports from other crates:
use ndarray::{arr1, Array1, ArrayView1, ArrayView2, ArrayViewMut1};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1};
use opencl::{DeviceType, OpenCLCtx};
use pyo3::prelude::*;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

// Imports from this lib:
use raster::StrideOrder::RowMajor;
use raster::{counts_to_strides, idx_to_sub, sub_to_idx, Ix};
use simulation::{CellType, Simulation};
use utils::vmod_nd;

pub mod fields;
pub mod opencl;
pub mod raster;
pub mod simulation;
pub mod utils;
pub mod vect_d;
pub mod vect_s;
pub mod velocities;

/// The implemented OpenCL kernels differ in whether they update the velocity or not.
/// [`EqnType::NavierStokes`] does, whereas [`EqnType::AdvectionDiffusion`] does not.
pub enum EqnType {
    NavierStokes,
    AdvectionDiffusion,
}

/// Calculate both the curl and Q-criterion in 2D.
///
/// All quantities are in lattice units.
///
/// NOTE: Assumes wall velocity is zero.
/// TODO: openCL version
fn calc_curl_qcrit_2d(
    vel: &ArrayView2<f32>,
    cells: &ArrayView1<i32>,
    counts: &ArrayView1<impl Into<Ix> + Copy>,
    curl: &mut ArrayViewMut1<f32>,
    qcrit: &mut ArrayViewMut1<f32>,
) {
    assert_eq!(vel.dim().0, cells.dim());

    fn is_wall(c: i32) -> f32 {
        ((c == (CellType::Wall as i32)) as i32) as f32
    }

    let counts = counts.to_owned().mapv(|x| x.into());
    let strides = &counts_to_strides(&counts, RowMajor);

    let offset = |sub: &Array1<Ix>, off: [Ix; 2]| {
        let off = arr1(&off);
        sub_to_idx(&vmod_nd(sub + off, &counts), &strides)
    };

    let curl_ = curl.as_slice_mut().unwrap();
    let qcrit_ = qcrit.as_slice_mut().unwrap();

    (curl_, qcrit_)
        .into_par_iter()
        .enumerate()
        .for_each(|(idx, (c, q))| {
            let sub = idx_to_sub(idx, &strides, RowMajor);

            let idx_x1 = offset(&sub, [-1, 0]);
            let idx_x2 = offset(&sub, [1, 0]);
            let idx_y1 = offset(&sub, [0, -1]);
            let idx_y2 = offset(&sub, [0, 1]);

            let dxux1: f32 = vel[(idx_x1, 0)] * (1.0 - is_wall(cells[idx_x1]));
            let dxux2: f32 = vel[(idx_x2, 0)] * (1.0 - is_wall(cells[idx_x2]));
            let dxuy1: f32 = vel[(idx_x1, 1)] * (1.0 - is_wall(cells[idx_x1]));
            let dxuy2: f32 = vel[(idx_x2, 1)] * (1.0 - is_wall(cells[idx_x2]));
            let dyux1: f32 = vel[(idx_y1, 0)] * (1.0 - is_wall(cells[idx_y1]));
            let dyux2: f32 = vel[(idx_y2, 0)] * (1.0 - is_wall(cells[idx_y2]));
            let dyuy1: f32 = vel[(idx_y1, 1)] * (1.0 - is_wall(cells[idx_y1]));
            let dyuy2: f32 = vel[(idx_y2, 1)] * (1.0 - is_wall(cells[idx_y2]));

            let dxux = (dxux2 - dxux1) / 2.0;
            let dxuy = (dxuy2 - dxuy1) / 2.0;
            let dyux = (dyux2 - dyux1) / 2.0;
            let dyuy = (dyuy2 - dyuy1) / 2.0;

            *c = dxuy - dyux;
            *q = dxux * dxux + dyuy * dyuy + 2.0 * dxuy * dyux;
        });
}

#[pyclass(name = "Fluid")]
struct FluidPy {
    sim: Arc<Mutex<Simulation>>,
}

#[pymethods]
impl FluidPy {
    #[getter]
    fn f<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray2<f32>> {
        let borrow = this.borrow();
        let array = &borrow.sim.lock().unwrap().fluid.f.host;
        unsafe { PyArray2::borrow_from_array(array, this.into_any()) }
    }

    #[getter]
    fn rho<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray1<f32>> {
        let borrow = this.borrow();
        let array = &borrow.sim.lock().unwrap().fluid.rho.host;
        unsafe { PyArray1::borrow_from_array(array, this.into_any()) }
    }

    #[getter]
    fn vel<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray2<f32>> {
        let borrow = this.borrow();
        let array = &borrow.sim.lock().unwrap().fluid.vel.host;
        unsafe { PyArray2::borrow_from_array(array, this.into_any()) }
    }

    #[getter]
    fn size_bytes(&self) -> usize {
        self.sim.lock().unwrap().fluid.size_bytes()
    }
}

#[pyclass(name = "Scalar")]
struct ScalarPy {
    sim: Arc<Mutex<Simulation>>,

    /// The index of the [`Scalar`] object inside the [`Simulation::tracers`] array.
    index: usize,
}

#[pymethods]
impl ScalarPy {
    #[getter]
    fn g<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray2<f32>> {
        let borrow = this.borrow();
        let sim = borrow.sim.lock().unwrap();
        if let Some(tracer) = sim.tracers.get(borrow.index) {
            let array = &tracer.g.host;
            unsafe { PyArray2::borrow_from_array(array, this.into_any()) }
        } else {
            // Just panic here because we shouldn't be able to get the [`ScalarPy`]
            // object if the [`Simulation`] doesn't have any tracers configured.
            panic!("No tracers configured")
        }
    }

    #[getter]
    fn val<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray1<f32>> {
        let borrow = this.borrow();
        let sim = borrow.sim.lock().unwrap();
        if let Some(tracer) = sim.tracers.get(borrow.index) {
            let array = &tracer.C.host;
            unsafe { PyArray1::borrow_from_array(array, this.into_any()) }
        } else {
            // Just panic here because we shouldn't be able to get the [`ScalarPy`]
            // object if the [`Simulation`] doesn't have any tracers configured.
            panic!("No tracers configured")
        }
    }

    #[getter]
    fn size_bytes(&self) -> usize {
        self.sim
            .lock()
            .unwrap()
            .tracers
            .get(self.index)
            .unwrap()
            .size_bytes()
    }
}

#[pyclass(name = "Cells")]
struct CellsPy {
    sim: Arc<Mutex<Simulation>>,
}

#[pymethods]
impl CellsPy {
    #[getter]
    fn cell_type<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray1<i32>> {
        let borrow = this.borrow();
        let sim = borrow.sim.lock().unwrap();
        let array = &sim.cells.typ.host;
        unsafe { PyArray1::borrow_from_array(array, this.into_any()) }
    }

    #[getter]
    fn size_bytes(&self) -> usize {
        self.sim.lock().unwrap().cells.size_bytes()
    }
}

#[pyclass(eq, eq_int, name = "DeviceType")]
#[derive(PartialEq)]
enum DeviceTypePy {
    CPU,
    GPU,
}

impl From<String> for DeviceTypePy {
    fn from(s: String) -> Self {
        match s.to_uppercase().as_str() {
            "CPU" => DeviceTypePy::CPU,
            "GPU" => DeviceTypePy::GPU,
            _ => panic!("Invalid device type. Expected either 'cpu' or 'gpu'."),
        }
    }
}

impl Into<DeviceType> for DeviceTypePy {
    fn into(self) -> DeviceType {
        match self {
            DeviceTypePy::CPU => DeviceType::CPU,
            DeviceTypePy::GPU => DeviceType::GPU,
        }
    }
}

impl Into<OpenCLCtx> for DeviceTypePy {
    fn into(self) -> OpenCLCtx {
        OpenCLCtx::new(self.into())
    }
}

/// Wrap the core [`Simulation`] class in an [`Arc`] + [`Mutex`] so it's threadsafe for Python.
#[pyclass(name = "Simulation")]
struct SimulationPy {
    sim: Arc<Mutex<Simulation>>,
}

impl SimulationPy {
    fn sim(&self) -> MutexGuard<'_, Simulation> {
        self.sim.lock().expect("Acquiring simulation mutex failed.")
    }
}

#[pymethods]
impl SimulationPy {
    #[new]
    fn new(dev: String, counts: PyReadonlyArray1<i32>, q: usize, omega_ns: f32) -> Self {
        let dev = DeviceTypePy::from(dev).into();
        let counts = counts.as_array().to_owned();
        Self {
            sim: Arc::new(Mutex::new(Simulation::new(dev, counts, q, omega_ns))),
        }
    }

    /// Get the total memory usage of the underlying [`Simulation`] object as seen on the GPU.
    #[getter]
    fn size_bytes(&self) -> usize {
        let sim = self.sim();
        let mut size_bytes = sim.fluid.size_bytes() + sim.cells.size_bytes();
        for tracer in sim.tracers.iter() {
            size_bytes += tracer.size_bytes();
        }
        size_bytes
    }

    #[getter]
    fn iteration(&self) -> u64 {
        self.sim().iteration
    }

    fn iterate(&mut self, iters: usize) {
        self.sim().iterate(iters)
    }

    fn set_gravity(&mut self, gravity: PyReadonlyArray1<f32>) {
        let gravity = gravity.as_array().to_owned();
        self.sim().set_gravity(gravity);
    }

    /// Add a scalar field which will follow the [Advection-Diffusion equation](https://en.wikipedia.org/wiki/Convection%E2%80%93diffusion_equation).
    ///
    /// Returns the [`ScalarPy`] object for the added tracer; you must keep this around to access the data.
    fn add_tracer<'py>(
        this: Bound<'py, Self>,
        q: usize,
        omega_ad: f32,
    ) -> PyResult<Bound<'py, ScalarPy>> {
        let this = this.borrow_mut();
        let mut sim = this.sim();
        let c = &sim.cells.counts;
        let tracer = Scalar::new(&sim.opencl, c, q, omega_ad);
        sim.add_tracer(tracer);

        let bound = Bound::new(
            this.py(),
            ScalarPy {
                sim: this.sim.clone(),
                index: sim.tracers.len() - 1,
            },
        )
        .expect("Bound::new ScalarPy failed.");

        Ok(bound)
    }

    /// Get a scalar field by index, which as previously been added.
    ///
    /// TODO: would be nice to make this by name or something (e.g. store a map of [`Scalar`]s not just a vec).
    fn get_tracer<'py>(this: Bound<'py, Self>, idx: usize) -> PyResult<Bound<'py, ScalarPy>> {
        let this = this.borrow();
        let bound = Bound::new(
            this.py(),
            ScalarPy {
                sim: this.sim.clone(),
                index: idx,
            },
        )
        .expect("Bound::new ScalarPy failed.");

        Ok(bound)
    }

    #[getter]
    fn cells<'py>(this: Bound<'py, Self>) -> Bound<'py, CellsPy> {
        let this = this.borrow();
        Bound::new(
            this.py(),
            CellsPy {
                sim: this.sim.clone(),
            },
        )
        .expect("Bound::new CellsPy failed.")
    }

    #[getter]
    fn fluid<'py>(this: Bound<'py, Self>) -> Bound<'py, FluidPy> {
        let this = this.borrow();
        Bound::new(
            this.py(),
            FluidPy {
                sim: this.sim.clone(),
            },
        )
        .expect("Bound::new FluidPy failed.")
    }

    fn write_checkpoint(&self, path: String) -> PyResult<()> {
        self.sim().write_checkpoint(path)?;
        Ok(())
    }

    #[staticmethod]
    fn load_checkpoint(dev: String, path: String) -> PyResult<SimulationPy> {
        let dev = DeviceTypePy::from(dev).into();
        let sim = Simulation::load_checkpoint(dev, path)?;
        let sim = Self {
            sim: Arc::new(Mutex::new(sim)),
        };
        Ok(sim)
    }
}

/// A module for configuring and running lattice Boltzmann simulations from Python.
#[pymodule]
fn boltzmann_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(name = "calc_curl_2d")]
    fn calc_curl_2d_py<'py>(
        _py: Python<'py>,
        vel: PyReadonlyArray2<f32>,
        cells: PyReadonlyArray1<i32>,
        counts: PyReadonlyArray1<Ix>,
        mut curl: PyReadwriteArray1<f32>,
        mut qcrit: PyReadwriteArray1<f32>,
    ) {
        calc_curl_qcrit_2d(
            &vel.as_array(),
            &cells.as_array(),
            &counts.as_array(),
            &mut curl.as_array_mut(),
            &mut qcrit.as_array_mut(),
        );
    }

    m.add_class::<SimulationPy>()?;
    m.add_class::<FluidPy>()?;
    m.add_class::<ScalarPy>()?;
    m.add_class::<CellsPy>()?;
    m.add_class::<DeviceTypePy>()?;

    Ok(())
}
