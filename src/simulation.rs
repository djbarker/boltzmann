use std::io::{BufReader, BufWriter};
use std::path::Path;

use ndarray::{arr1, Array1};
use opencl3::command_queue::CommandQueue;
use opencl3::kernel::ExecuteKernel;
use opencl3::types::cl_event;
use rmp_serde::{Deserializer, Serializer};
use serde::{Deserialize, Serialize};

use crate::fields::{Fluid, FluidDeserializer, MemUsage, Scalar, ScalarDeserializer};
use crate::opencl::{CtxDeserializer, DataNd, DataNdDeserializer};
use crate::opencl::{Data, Data2d, OpenCLCtx};

/// Contains information about the cells.
/// For example, whether they are a fluid or wall, have fixed velocity, etc.
#[derive(Serialize)]
pub struct Cells {
    /// The flags of the cell as given by [`CellType`].
    /// TODO: rename `typ` -> `flags`.
    pub typ: DataNd<i32>,

    /// Cell count in each dimension.
    pub counts: Array1<usize>,
}

impl Cells {
    pub fn new(opencl: &OpenCLCtx, counts: &[usize]) -> Self {
        let mut out = Self {
            typ: Data::new(opencl, counts, 0),
            counts: arr1(counts),
        };

        out.write_to_dev(opencl);

        out
    }

    // Copy data from our host arrays into the OpenCL buffers.
    pub fn write_to_dev(&mut self, opencl: &OpenCLCtx) {
        let queue = &opencl.queue;
        let _write_t = self.typ.enqueue_write(queue, "typ");

        queue
            .finish()
            .expect("queue.finish() failed [Cells::write_to_dev]");
    }
}

impl MemUsage for Cells {
    fn size_bytes(&self) -> usize {
        std::mem::size_of::<i32>() * (self.typ.host.len())
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
            typ: self.typ.with_context(opencl),
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

/// Stores the [`Fluid`] and any associated [`Scalar`] fields.
/// Stores the [`OpenCLCtx`] and Calls the OpenCL kernels.
#[derive(Serialize)]
pub struct Simulation {
    #[serde(skip)]
    pub(crate) opencl: OpenCLCtx,
    pub cells: Cells,
    pub fluid: Fluid,
    pub tracers: Vec<Scalar>,
    pub gravity: Array1<f32>,
    pub iteration: u64,
}

#[derive(Deserialize)]
struct SimulationDeserializer {
    cells: CellsDeserializer,
    fluid: FluidDeserializer,
    tracers: Vec<ScalarDeserializer>,
    gravity: Array1<f32>,
    iteration: u64,
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
                .map(|t| t.with_context(&opencl))
                .collect(),
            gravity: self.gravity,
            iteration: self.iteration,
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
            tracers: Vec::new(),
            gravity: Array1::zeros([counts.len()]),
            iteration: 0,
        }
    }

    pub fn set_gravity(&mut self, gravity: Array1<f32>) {
        self.gravity = gravity;
    }

    pub fn add_tracer(&mut self, tracer: Scalar) {
        self.tracers.push(tracer);
    }

    /// Calculate equilibrium distribution functions from macroscopic variables.
    ///
    /// Once we've initialized the density, velocity, etc arrays in Python we must call this
    /// before iterating so that the distribution function arrays are populated.
    /// If we are loading from a checkpoint we do not need to call this.
    /// We also need to call [`Simulation::finalize`] to copy the data to the OpenCL device.
    fn equilibrate(&mut self) {
        self.fluid.equilibrate();
        for tracer in self.tracers.iter_mut() {
            tracer.equilibrate(self.fluid.vel.host.view());
        }
    }

    /// Copy data to OpenCL buffers.
    ///
    /// We must call this before iterating so that the data is on the OpenCL device.
    fn finalize(&mut self) {
        self.cells.write_to_dev(&self.opencl);
        self.fluid.write_to_dev(&self.opencl);
        for tracer in self.tracers.iter_mut() {
            tracer.write_to_dev(&self.opencl);
        }
    }

    /// See: opencl3 example [`basic.rs`](https://github.com/kenba/opencl3/blob/main/examples/basic.rs)
    pub fn iterate(&mut self, iters: usize) {
        if self.iteration == 0 {
            self.equilibrate();
            self.finalize();
        }

        if iters % 2 != 0 {
            panic!("iters must be even")
        }

        let queue: &CommandQueue = &self.opencl.queue;

        // Round global work size upto the next even multiple of 8.
        let count_s = self.cells.counts.mapv(|c| {
            let r = c % 16;
            c + (16 - r)
        });
        let count_s = count_s.as_slice().unwrap();

        // I haven't extensively optimized this but 8x8 seems like a good value for 2D.
        let wsize = [8, 8];

        // TODO: make this generic over [`D`], [`Q`] and [`EqnType`]!
        let d2q9 = &self.opencl.d2q9_ns_kernel;
        let d2q5 = &self.opencl.d2q5_ad_kernel;

        // TODO: don't copy this every iteration batch
        let mut qs_d2q9 = Data2d::from_host(&self.opencl, self.fluid.model.qs.clone());
        let mut qs_d2q5 = Data2d::from_host(&self.opencl, self.fluid.model.qs.clone());

        qs_d2q9.enqueue_write(queue, "qs_d2q9");
        qs_d2q5.enqueue_write(queue, "qs_d2q5");

        let mut events: Vec<cl_event> = Vec::default();

        macro_rules! _exec_kernel_inner {
            // Base case.
            ($kernel:expr) => {
                $kernel
            };
            // Almost-base case.
            // This seems unnecessary; why not just do this in the recurse case and rely on the base case?
            // But it _is_ needed, see: https://users.rust-lang.org/t/tail-recursive-macros/905/3.
            ($kernel:expr, $arg:expr) => {
                $kernel.set_arg($arg)
            };
            // Recurse case.
            ($builder:expr, $arg:expr, $($args:expr),+) => {
                _exec_kernel_inner!{
                    _exec_kernel_inner!{$builder, $arg}
                , $($args),*}
            };
        }

        /// Run the specified kernel with the given arguments.
        ///
        /// Translates the given arguments into the correct sequence of [`ExecuteKernel::set_arg`] calls,
        /// specifies the work size, and runs the kernel.
        macro_rules! exec_kernel {
            // Entry case.
            ($kernel:expr, $($args:expr),*) => {
                _exec_kernel_inner!{ ExecuteKernel::new($kernel), $($args),*}
                    .set_arg(&(self.cells.counts[0] as i32))
                    .set_arg(&(self.cells.counts[1] as i32))
                    .set_local_work_sizes(&wsize)
                    .set_global_work_sizes(count_s)
                    .enqueue_nd_range(&queue)
                    .expect("ExecuteKernel::new failed.")
            };
        }

        for iter in 0..iters {
            // We cannot pass bools to OpenCL so get an interger with values 0 or 1.
            let even = (1 - iter % 2) as i32;

            let d2q9_event = unsafe {
                exec_kernel!(
                    d2q9,
                    &even,
                    &self.fluid.omega,
                    &self.gravity[0],
                    &self.gravity[1],
                    &mut self.fluid.f.dev,
                    &mut self.fluid.rho.dev,
                    &mut self.fluid.vel.dev,
                    &self.cells.typ.dev,
                    &qs_d2q9.dev
                )
            };

            events.push(d2q9_event.get());
            queue.finish().expect("queue.finish failed");

            for tracer in self.tracers.iter_mut() {
                let d2q5_event = unsafe {
                    exec_kernel!(
                        d2q5,
                        &even,
                        &tracer.omega,
                        &mut tracer.g.dev,
                        &mut tracer.C.dev,
                        &self.fluid.vel.dev,
                        &self.cells.typ.dev,
                        &qs_d2q5.dev
                    )
                };

                events.push(d2q5_event.get());
                queue.finish().expect("queue.finish failed");
            }
        }

        // Read the data back from the OpenCL device buffers to the host arrays.
        self.fluid.read_to_host(&self.opencl);
        for tracer in self.tracers.iter_mut() {
            tracer.read_to_host(&self.opencl);
        }

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
