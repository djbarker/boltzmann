use std::collections::HashMap;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use ndarray::{arr1, Array1};
use opencl3::command_queue::CommandQueue;
use opencl3::kernel::{ExecuteKernel, Kernel};
use rmp_serde::{Deserializer, Serializer};
use serde::{Deserialize, Serialize};

use crate::fields::{
    make_kernel_key, Field, Fluid, FluidDeserializer, MemUsage, Scalar, ScalarDeserializer,
};
use crate::opencl::{CtxDeserializer, Data1d, Data1dDeserializer, DataNd, DataNdDeserializer};
use crate::opencl::{Data, OpenCLCtx};

/// Contains information about the cells.
/// For example, whether they are a fluid or wall, have fixed velocity, etc.
#[derive(Serialize)]
pub struct Cells {
    /// The flags of the cell as given by [`CellType`].
    pub flags: DataNd<i32>,

    /// Cell count in each dimension.
    pub counts: Array1<usize>,
}

impl Cells {
    pub fn new(opencl: &OpenCLCtx, counts: &[usize]) -> Self {
        let mut out = Self {
            // TODO: here & in deserializer could be read-only
            flags: Data::from_val_rw(opencl, counts, 0),
            counts: arr1(counts),
        };

        out.write_to_dev(opencl);

        out
    }

    // Copy data from our host arrays into the OpenCL buffers.
    pub fn write_to_dev(&mut self, opencl: &OpenCLCtx) {
        let queue = &opencl.queue;
        let _write_t = self.flags.enqueue_write(queue, "typ");

        queue
            .finish()
            .expect("queue.finish() failed [Cells::write_to_dev]");
    }
}

impl MemUsage for Cells {
    fn size_bytes(&self) -> usize {
        std::mem::size_of::<i32>() * (self.flags.host.len())
    }
}

/// See [`FluidDeserializer`]
#[derive(Deserialize)]
struct CellsDeserializer {
    typ: DataNdDeserializer<i32>,
    counts: Array1<usize>,
}

impl CtxDeserializer for CellsDeserializer {
    type Target = Cells;

    fn with_context(self, opencl: &OpenCLCtx) -> Self::Target {
        Self::Target {
            flags: self.typ.with_context(opencl),
            counts: self.counts,
        }
    }
}

/// Flags controlling the cell behaviours.
#[repr(C)]
pub enum CellType {
    Fluid = 0,
    Wall = 1,
    FixedFluidVelocity = 2,
    FixedFluidDensity = 4,
    /// == FIXED_FLUID_VELOCITY | FIXED_FLUID_DENSITY
    FixedFluid = 6,
    FixedScalarValue = 8,
}

/// Couple a tracer to the fluid via the Boussinesq approximation.
#[derive(Serialize)]
pub struct BoussinesqCoupling {
    /// Tracer field to couple the fluid to.
    pub tracer: String,

    /// Coupling strength.
    pub alpha: f32,

    /// Value of the tracer field which corresponds to zero buoyancy.
    pub c0: f32,

    /// Gravity vector.
    pub gravity: Data1d<f32>,
}

#[derive(Deserialize)]
pub struct BoussinesqCouplingDeserializer {
    tracer: String,
    alpha: f32,
    c0: f32,
    gravity: Array1<f32>,
}

impl CtxDeserializer for BoussinesqCouplingDeserializer {
    type Target = BoussinesqCoupling;

    fn with_context(self, opencl: &OpenCLCtx) -> Self::Target {
        Self::Target {
            tracer: self.tracer,
            alpha: self.alpha,
            c0: self.c0,
            gravity: Data1d::from_host_ro(opencl, self.gravity),
        }
    }
}

/// Stores the [`Fluid`] and any associated [`Scalar`] fields.
/// Stores the [`OpenCLCtx`] and Calls the OpenCL kernels.
#[derive(Serialize)]
pub struct Simulation {
    #[serde(skip)]
    pub(crate) opencl: OpenCLCtx,
    pub cells: Cells,
    pub fluid: Fluid,
    pub tracers: HashMap<String, Scalar>,
    pub gravity: Option<Data1d<f32>>,
    pub iteration: u64,
    pub couplings: Vec<BoussinesqCoupling>,
}

#[derive(Deserialize)]
struct SimulationDeserializer {
    cells: CellsDeserializer,
    fluid: FluidDeserializer,
    tracers: HashMap<String, ScalarDeserializer>,
    gravity: Option<Data1dDeserializer<f32>>,
    iteration: u64,
    couplings: Vec<BoussinesqCouplingDeserializer>,
}

impl SimulationDeserializer {
    /// # Note
    ///
    /// This is not an implementation of [`CtxDeserializer`] because it's the top level and
    /// we must consume the [`OpenCLCtx`] to create the [`Simulation`] object.
    fn to_simulation(self, opencl: OpenCLCtx) -> Simulation {
        Simulation {
            cells: self.cells.with_context(&opencl),
            fluid: self.fluid.with_context(&opencl),
            tracers: self
                .tracers
                .into_iter()
                .map(|(n, t)| (n, t.with_context(&opencl)))
                .collect(),
            gravity: self.gravity.map(|g| g.with_context(&opencl)),
            iteration: self.iteration,
            couplings: self
                .couplings
                .into_iter()
                .map(|c| c.with_context(&opencl))
                .collect(),
            opencl: opencl,
        }
    }
}

impl Simulation {
    pub fn new(opencl: OpenCLCtx, counts: &[usize], q: usize, omega_ns: f32) -> Self {
        let cells = Cells::new(&opencl, counts);
        let fluid = Fluid::new(&opencl, counts, q, omega_ns);

        Self {
            opencl: opencl,
            cells: cells,
            fluid: fluid,
            tracers: HashMap::new(),
            gravity: None,
            iteration: 0,
            couplings: Vec::new(),
        }
    }

    pub fn set_gravity(&mut self, gravity: Array1<f32>) {
        self.gravity = Some(Data1d::from_host_ro(&self.opencl, gravity));
    }

    /// Add a coupling to the named [`Scalar`] field.
    ///
    /// Panics if the name is unknown.
    pub fn add_boussinesq_coupling(
        &mut self,
        tracer: impl AsRef<str>,
        alpha: f32,
        c0: f32,
        gravity: Array1<f32>,
    ) {
        assert!(self.tracers.contains_key(tracer.as_ref()));

        self.couplings.push(BoussinesqCoupling {
            tracer: tracer.as_ref().to_owned(),
            alpha: alpha,
            c0: c0,
            gravity: Data1d::from_host_ro(&self.opencl, gravity),
        });
    }

    /// Add a [`Scalar`] field with the given name.
    ///
    /// Panics if the name already exists.
    pub fn add_tracer(&mut self, name: impl AsRef<str>, tracer: Scalar) {
        if self
            .tracers
            .insert(name.as_ref().to_owned(), tracer)
            .is_some()
        {
            panic!("Existing tracer '{}'", name.as_ref());
        }
    }

    /// Return a previously added [`Scalar`] field.
    ///
    /// Returns [`Some<Scalar>`] if the field exists, otherwise [`None`].
    pub fn get_tracer(&self, name: impl AsRef<str>) -> Option<&Scalar> {
        self.tracers.get(name.as_ref())
    }

    /// Calculate equilibrium distribution functions from macroscopic variables.
    ///
    /// Once we've initialized the density, velocity, etc arrays in Python we must call this
    /// before iterating so that the distribution function arrays are populated.
    /// If we are loading from a checkpoint we do not need to call this.
    /// We also need to call [`Simulation::finalize`] to copy the data to the OpenCL device.
    fn equilibrate(&mut self) {
        self.fluid.equilibrate();
        for (_, tracer) in self.tracers.iter_mut() {
            tracer.equilibrate(self.fluid.vel.host.view());
        }
    }

    /// Copy data to OpenCL buffers.
    ///
    /// We must call this before iterating so that the data is on the OpenCL device.
    fn finalize(&mut self) {
        self.cells.write_to_dev(&self.opencl);
        self.fluid.write_to_dev(&self.opencl);
        for (_, tracer) in self.tracers.iter_mut() {
            tracer.write_to_dev(&self.opencl);
        }

        if let Some(gravity) = &mut self.gravity {
            gravity.enqueue_write(&self.opencl.queue, "gravity");
        }

        for coupling in self.couplings.iter_mut() {
            coupling
                .gravity
                .enqueue_write(&self.opencl.queue, "coupling.gravity");
        }

        self.opencl
            .queue
            .finish()
            .expect("queue.finish() failed [Simulation::finalize]");
    }

    /// See: opencl3 example [`basic.rs`](https://github.com/kenba/opencl3/blob/main/examples/basic.rs)
    pub fn iterate(&mut self, iters: usize) {
        if iters % 2 != 0 {
            panic!("iters must be even")
        }

        if self.iteration == 0 {
            self.equilibrate();
            self.finalize();
        }

        // Allocate the acceleration buffer if it's needed and not already allocated.
        if self.gravity.is_some() || !self.couplings.is_empty() {
            if self.fluid.acc.is_none() {
                let n = self.cells.counts.product();
                let d = self.cells.counts.len();
                self.fluid.acc = Some(DataNd::from_val_rw(&self.opencl, vec![n, d], 0.0));
            }
        }

        let queue: &CommandQueue = &self.opencl.queue;

        // Round global work size up to the next multiple of 16.
        let count_s = self.cells.counts.mapv(|c| {
            let r = c % 16;
            c + (16 - r)
        });
        let count_s = count_s.as_slice().unwrap();

        // I haven't extensively optimized this but 16x16 seems like a good value for 2D for my GPU.
        // According `clinfo` the max work group size is 1024, which is 32 x 32.
        let wsize = match self.fluid.model.D() {
            1 => Vec::from([1024]),
            2 => Vec::from([16, 16]),
            3 => Vec::from([8, 8, 8]),
            _ => panic!("D must be 1, 2 or 3"),
        };

        fn get_kernel<'a>(opencl: &'a OpenCLCtx, f: &impl Field) -> &'a Kernel {
            let key = make_kernel_key(f);
            let msg = format!("Kernel not found: {:?}", key);
            opencl.kernels.get(&key).expect(msg.as_str())
        }

        // Things which are constant in the kernel call, but still need to be copied over
        // TODO: only do this once
        // TODO: make read-only and use __constant specifier in the kernel args
        let mut s = Data1d::from_host_ro(&self.opencl, self.cells.counts.mapv(|c| c as i32));
        s.enqueue_write(&queue, "s");

        if let Some(gravity) = &self.gravity {
            let acc = self.fluid.acc.as_mut().expect("no acceleration buffer");

            // unsafe {
            //     ExecuteKernel::new(&self.opencl.set_grav_2d)
            //         .set_arg(&s.dev)
            //         .set_arg(&mut acc.dev)
            //         .set_arg(&gravity.dev)
            //         .set_local_work_sizes(&wsize)
            //         .set_global_work_sizes(count_s)
            //         .enqueue_nd_range(&queue)
            //         .expect("ExecuteKernel::new failed.")
            // };
        };

        // See comment below about passing a null ptr.
        let mut dummy = DataNd::<f32>::from_val_rw(&self.opencl, vec![1], 0.0);
        dummy.enqueue_write(queue, "dummy");

        for iter in 0..iters {
            // We cannot pass bools to OpenCL so get an integer with values 0 or 1.
            let even = (1 - iter % 2) as i32;

            // Set source terms for the acceleration for Boussinesq approximation.
            for coupling in self.couplings.iter() {
                let acc = self.fluid.acc.as_mut().expect("no acceleration buffer");

                let tracer = self.tracers.get(&coupling.tracer).unwrap();
                unsafe {
                    ExecuteKernel::new(&self.opencl.set_bous_2d)
                        .set_arg(&s.dev)
                        .set_arg(&mut acc.dev)
                        // .set_arg(&tracer.C.dev)
                        .set_arg(&coupling.alpha)
                        .set_arg(&coupling.c0)
                        .set_arg(&coupling.gravity.dev)
                        .set_local_work_sizes(&wsize)
                        .set_global_work_sizes(count_s)
                        .enqueue_nd_range(&queue)
                        .expect("ExecuteKernel::new failed.")
                };
            }

            let fluid_kernel = get_kernel(&self.opencl, &self.fluid);

            // I have not found a good way with the rust OpenCL crate to pass a null ptr.
            // Here we instead pass in a dummy buffer and use a flag to determine
            // whether to use it or not.
            let (acc, use_acc) = if let Some(acc) = &mut self.fluid.acc {
                (acc, 1)
            } else {
                (&mut dummy, 0)
            };

            unsafe {
                ExecuteKernel::new(fluid_kernel)
                    .set_arg(&s.dev)
                    .set_arg(&even)
                    .set_arg(&self.fluid.omega)
                    .set_arg(&mut self.fluid.f.dev)
                    .set_arg(&mut self.fluid.rho.dev)
                    .set_arg(&mut self.fluid.vel.dev)
                    .set_arg(&mut acc.dev)
                    .set_arg(&use_acc)
                    .set_arg(&self.cells.flags.dev)
                    .set_local_work_sizes(&wsize)
                    .set_global_work_sizes(count_s)
                    .enqueue_nd_range(&queue)
                    .expect("ExecuteKernel::new failed.")
            };

            for (_, tracer) in self.tracers.iter_mut() {
                unsafe {
                    ExecuteKernel::new(get_kernel(&self.opencl, tracer))
                        .set_arg(&s.dev)
                        .set_arg(&even)
                        .set_arg(&tracer.omega)
                        .set_arg(&mut tracer.g.dev)
                        .set_arg(&mut tracer.C.dev)
                        .set_arg(&self.fluid.vel.dev)
                        .set_arg(&self.cells.flags.dev)
                        .set_local_work_sizes(&wsize)
                        .set_global_work_sizes(count_s)
                        .enqueue_nd_range(&queue)
                        .expect("ExecuteKernel::new failed.")
                };
            }
        }

        queue.finish().expect("queue.finish failed");

        // Read the data back from the OpenCL device buffers to the host arrays.
        self.fluid.read_to_host(&self.opencl);
        for (_, tracer) in self.tracers.iter_mut() {
            tracer.read_to_host(&self.opencl);
        }

        queue.finish().expect("queue.finish()");

        self.iteration += iters as u64;
    }

    pub fn write_checkpoint(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let file = std::fs::File::create(path)?;
        let mut buff = BufWriter::with_capacity(1024 * 1024, file); // 1 mebibyte buffer
        let mut serializer = Serializer::new(&mut buff);
        self.serialize(&mut serializer).unwrap();

        Ok(())
    }

    pub fn load_checkpoint(
        opencl: OpenCLCtx,
        path: impl AsRef<Path>,
    ) -> std::io::Result<Simulation> {
        let file = std::fs::File::open(path)?;
        let buff = BufReader::with_capacity(1024 * 1024, file);
        let mut de = Deserializer::new(buff);
        let sim = SimulationDeserializer::deserialize(&mut de).unwrap();
        let mut sim = sim.to_simulation(opencl);

        // In general iteration is not zero, so finalize here to ensure the data is copied to the device.
        // If we checkpointed at iteration zero this will happen twice but, meh.
        sim.finalize();

        Ok(sim)
    }
}
