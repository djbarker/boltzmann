use std::fmt::format;
// Std imports:
use std::sync::{Arc, Mutex, MutexGuard};
use std::usize;

// Imports from other crates:
use numpy::{Ix1, PyArray, PyArrayDyn, PyReadonlyArray1};
use opencl::DeviceType;
use opencl3::device::{CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU};
use pyo3::prelude::*;

// Imports from this lib:
use fields::{MemUsage, Omega, Scalar};
use simulation::Simulation;

pub mod fields;
pub mod opencl;
pub mod raster;
pub mod simulation;
pub mod utils;
pub mod velocities;

#[pyclass(name = "Omega")]
#[derive(Clone, Copy)]
enum OmegaPy {
    BGK(f32),
    TRT(f32, f32),
}

impl OmegaPy {
    pub fn __repr__(&self) -> String {
        match self {
            OmegaPy::BGK(o) => format!("BGK({})", o),
            OmegaPy::TRT(op, on) => format!("TRT({}, {})", op, on),
        }
    }
}

impl From<OmegaPy> for Omega {
    fn from(value: OmegaPy) -> Self {
        match value {
            OmegaPy::BGK(omega) => Omega::BGK(omega),
            OmegaPy::TRT(omega_pos, omega_neg) => Omega::TRT(omega_pos, omega_neg),
        }
    }
}

/// Stores the density and velocity fields, relaxation parameter and a velocity set.
#[pyclass(name = "Fluid")]
struct FluidPy {
    sim: Arc<Mutex<Simulation>>,
}

#[pymethods]
impl FluidPy {
    /// The fluid density data.
    #[getter]
    fn rho<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArrayDyn<f32>> {
        let borrow = this.borrow();
        let array = &borrow.sim.lock().unwrap().fluid.rho.host;
        unsafe { PyArrayDyn::borrow_from_array(array, this.into_any()) }
    }

    /// The fluid velocity data.
    #[getter]
    fn vel<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArrayDyn<f32>> {
        let borrow = this.borrow();
        let array = &borrow.sim.lock().unwrap().fluid.vel.host;
        unsafe { PyArrayDyn::borrow_from_array(array, this.into_any()) }
    }

    /// The memory usage of the :py:class:`Fluid`` object.
    ///
    /// :returns: The total size in bytes.
    #[getter]
    fn size_bytes(&self) -> usize {
        self.sim.lock().unwrap().fluid.size_bytes()
    }
}

/// Stores the scalar field data, relaxation parameter and a velocity set.
#[pyclass(name = "Scalar")]
struct ScalarPy {
    sim: Arc<Mutex<Simulation>>,

    /// The name of the [`Scalar`] object inside the [`Simulation::tracers`] map.
    name: String,
}

#[pymethods]
impl ScalarPy {
    /// The scalar field data.
    #[getter]
    fn val<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArrayDyn<f32>> {
        let borrow = this.borrow();
        let sim = borrow.sim.lock().unwrap();
        if let Some(tracer) = sim.tracers.get(borrow.name.as_str()) {
            let array = &tracer.C.host;
            unsafe { PyArrayDyn::borrow_from_array(array, this.into_any()) }
        } else {
            // Just panic here because we shouldn't be able to get the [`ScalarPy`]
            // object if the [`Simulation`] doesn't have any tracers configured.
            panic!("No tracers configured")
        }
    }

    /// The memory usage of the :py:class:`Scalar`` object.
    ///
    /// :returns: The total size in bytes.
    #[getter]
    fn size_bytes(&self) -> usize {
        self.sim
            .lock()
            .unwrap()
            .tracers
            .get(self.name.as_str())
            .unwrap()
            .size_bytes()
    }
}

/// Stores the grid data, e.g. what flags are set for each cell and the size in each dimension.
#[pyclass(name = "Cells")]
struct CellsPy {
    sim: Arc<Mutex<Simulation>>,
}

#[pymethods]
impl CellsPy {
    /// The cell flag data.
    ///
    /// See :py:class:`~boltzmann.core.CellType` for the possible values.
    ///
    /// :returns: An array containing the flag value for each cell.
    #[getter]
    fn flags<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArrayDyn<i32>> {
        let borrow = this.borrow();
        let sim = borrow.sim.lock().unwrap();
        let array = &sim.cells.flags.host;
        unsafe { PyArrayDyn::borrow_from_array(array, this.into_any()) }
    }

    /// The memory usage of the :py:class:`Cells` object.
    ///
    /// :returns: The total size in bytes.
    #[getter]
    fn size_bytes(&self) -> usize {
        self.sim.lock().unwrap().cells.size_bytes()
    }

    /// The number of grid cells in each dimension.
    ///
    /// :returns: The count in each dimension.
    #[getter]
    fn counts<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray<usize, Ix1>> {
        let borrow = this.borrow();
        let sim = borrow.sim.lock().unwrap();
        let array = &sim.cells.counts;
        unsafe { PyArray::<usize, Ix1>::borrow_from_array(array, this.into_any()) }
    }

    /// The total number of grid cells.
    ///
    /// :returns: The product of the count in each dimension.
    #[getter]
    fn count(&self) -> usize {
        self.sim.lock().unwrap().cells.counts.product()
    }
}

/// The central simulation object which groups together information about the domain (:py:class:`Cells`),
/// the fluid (:py:class:`Fluid`) and any scalar fields (:py:class:`Scalar`).
#[pyclass(name = "Simulation")]
struct SimulationPy {
    /// Wrap the core rust [`Simulation`] class in an [`Arc`] + [`Mutex`] so it's threadsafe for Python.
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
    #[pyo3(signature = (dev, counts, omega, q=None))]
    fn new(dev: String, counts: Vec<usize>, omega: OmegaPy, q: Option<usize>) -> Self {
        // If q is not provided infer the most common kernels for each dimension.
        let q = q.unwrap_or(match counts.len() {
            1 => 3,
            2 => 9,
            3 => 27,
            _ => panic!("Invalid number of dimensions. Expected either 1, 2 or 3."),
        });

        let dev = DeviceType::from(dev).into();

        Self {
            sim: Arc::new(Mutex::new(Simulation::new(dev, &counts, q, omega.into()))),
        }
    }

    #[getter]
    fn device_info(&self) -> String {
        let device = &self.sim().opencl.device;
        let vendor = device.vendor().expect("Error getting OpenCL Vendor");
        let dev_type = match device.dev_type().expect("Error getting OpenCL Device Type") {
            CL_DEVICE_TYPE_GPU => "GPU",
            CL_DEVICE_TYPE_CPU => "CPU",
            _ => "UNKNOWN",
        };
        format!("{} {}", vendor, dev_type)
    }

    // #[getter]
    // fn test<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray2<f32>> {
    //     let borrow = this.borrow();
    //     let array = &borrow.test;
    //     let array = array.slice(s![1..100, 1..100]);
    //     unsafe { PyArray2::borrow_from_array(&array, this.into_any()) }
    // }

    /// Get the total memory usage of the underlying :py:class:`Simulation` object as seen on the GPU.
    ///
    /// :returns: The total size in bytes.
    #[getter]
    fn size_bytes(&self) -> usize {
        let sim = self.sim();
        let mut size_bytes = sim.fluid.size_bytes() + sim.cells.size_bytes();
        for (_, tracer) in sim.tracers.iter() {
            size_bytes += tracer.size_bytes();
        }
        size_bytes
    }

    /// The number of iterations that have been run so far.
    #[getter]
    fn iteration(&self) -> u64 {
        self.sim().iteration
    }

    /// Run the lattice Boltzmann simulation.
    ///
    /// :param iters: The number of simulation timesteps to run.
    fn iterate(&mut self, iters: usize) {
        self.sim().iterate(iters)
    }

    /// Set the body force that the fluid will feel.
    ///
    /// :param gravity: The body force vector in lattice units.
    fn set_gravity(&mut self, gravity: PyReadonlyArray1<f32>) {
        let gravity = gravity.as_array().to_owned();
        self.sim().set_gravity(gravity);
    }

    /// Adds a `Boussinesq approximation <https://en.wikipedia.org/wiki/Boussinesq_approximation_(buoyancy)>`_ coupling between the fluid and a scalar field.
    ///
    /// If the scalar field has a value denoted by :math:`c`, the acceleration is given by
    ///
    /// .. math::
    ///
    ///     \mathbf a = \alpha \mathbf g (c - c_0)
    ///
    /// .. note::
    ///     
    ///     The coupling constant :math:`\alpha` is not strictly needed as it could be absorbed into :math:`g`.
    ///     It is included for ease of specification where you probably already have a gravity vector in mind.
    ///     
    /// :param tracer: The :py:class:`Scalar` object to couple to.
    /// :param alpha: The coupling strength.
    /// :param c0: The value of the scalar field for which there is zero acceleration.
    /// :param g: The body force vector in lattice units.
    fn add_boussinesq_coupling<'py>(
        &mut self,
        tracer: Bound<'py, ScalarPy>,
        alpha: f32,
        c0: f32,
        g: PyReadonlyArray1<f32>,
    ) {
        let tracer = tracer.borrow();
        let g = g.as_array().to_owned();
        self.sim()
            .add_boussinesq_coupling(&tracer.name, alpha, c0, g);
    }

    /// Add a scalar field which will follow the `advection-diffusion equation <https://en.wikipedia.org/wiki/Convection%E2%80%93diffusion_equation>`_.
    ///
    /// Returns the :py:class:`Scalar` object for the added tracer; you must keep this around to access the data.
    /// This function is idempotent (over the name) and will return the existing tracer if it already exists.
    ///
    /// :param name: The unique name of the tracer.
    /// :param omega: The relaxation parameter for the advection-diffusion equation.
    /// :param q: The number of discrete velocities to use for the advection-diffusion equation kernel.
    /// :returns: The :py:class:`Scalar` object.
    #[pyo3(signature = (name, omega, q=None))]
    fn add_tracer<'py>(
        this: Bound<'py, Self>,
        name: String,
        omega: OmegaPy,
        q: Option<usize>,
    ) -> PyResult<Bound<'py, ScalarPy>> {
        let this = this.borrow_mut();
        let mut sim = this.sim();

        // Add the tracer if it does not exist.
        if sim.get_tracer(&name).is_none() {
            // If q is not provided infer the most common kernels for each dimension.
            let q = q.unwrap_or(match sim.cells.counts.len() {
                1 => 3,
                2 => 5,
                3 => 7,
                _ => panic!("Invalid number of dimensions. Expected either 1, 2 or 3."),
            });

            let c = sim.cells.counts.as_slice().unwrap();
            let tracer = Scalar::new(&sim.opencl, c, q, omega.into());
            sim.add_tracer(&name, tracer);
        }

        let bound = Bound::new(
            this.py(),
            ScalarPy {
                sim: this.sim.clone(),
                name: name.clone(),
            },
        )
        .expect("Bound::new ScalarPy failed.");

        Ok(bound)
    }

    /// Get a previously added :py:class:`Scalar` field by name.
    /// This will fail if the name is unknown.
    ///
    /// :param name: The unique name of the tracer to get.
    fn get_tracer<'py>(this: Bound<'py, Self>, name: String) -> PyResult<Bound<'py, ScalarPy>> {
        let this = this.borrow();
        let bound = Bound::new(
            this.py(),
            ScalarPy {
                sim: this.sim.clone(),
                name: name,
            },
        )
        .expect("Bound::new ScalarPy failed.");

        Ok(bound)
    }

    /// The :py:class:`Cells` object containing the domain info for the simulation.
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

    /// The :py:class:`Fluid` object containing the fluid density & velocity fields.
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

    /// Save a checkpoint in `MessagePack <https://msgpack.org/index.html>`_ format.
    ///
    /// :param path: Where to save the checkpoint.
    fn write_checkpoint(&self, path: String) -> PyResult<()> {
        self.sim().write_checkpoint(path)?;
        Ok(())
    }

    /// Construct a :py:class:`Simulation` by loading from a previously saved checkpoint.
    ///
    /// :param path: Where to load the checkpoint from.
    #[staticmethod]
    fn load_checkpoint(dev: String, path: String) -> PyResult<SimulationPy> {
        let dev = DeviceType::from(dev).into();
        let sim = Simulation::load_checkpoint(dev, path)?;
        let sim = Self {
            sim: Arc::new(Mutex::new(sim)),
        };
        Ok(sim)
    }
}

/// A module for configuring and running lattice Boltzmann simulations from Python.
#[pymodule]
fn boltzmann(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<OmegaPy>()?;
    m.add_class::<SimulationPy>()?;
    m.add_class::<FluidPy>()?;
    m.add_class::<ScalarPy>()?;
    m.add_class::<CellsPy>()?;

    Ok(())
}
