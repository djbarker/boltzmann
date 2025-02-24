use std::io::{BufReader, BufWriter};
use std::path::Path;

use ndarray::Array1;
use opencl3::command_queue::CommandQueue;
use opencl3::kernel::ExecuteKernel;
use opencl3::types::cl_event;
use rmp_serde::{Deserializer, Serializer};
use serde::{Deserialize, Serialize};

use crate::fields::{Fluid, FluidDeserializer, MemUsage, Scalar, ScalarDeserializer};
use crate::opencl::{CtxDeserializer, Data1dDeserializer, Data2dDeserializer};
use crate::raster::IxLike;
use crate::{
    opencl::{Data, Data1d, Data2d, OpenCLCtx},
    raster::Ix,
    velocities::VelocitySet,
};

/// Contains information about the cells.
/// For example, whether they are a fluid or wall, have fixed velocity, etc.
#[derive(Serialize)]
pub struct Cells {
    /// The type of the cell as given by [`CellType`].
    pub typ: Data1d<i32>,

    /// The _upstream_ cell indices in each direction.
    /// NOTE: Not public because it's basically an implementation detail.
    /// NOTE: Currently this gets serialized, we could recreate it but, meh; it's going anyway.
    idx: Data2d<i32>,

    /// Cell count in each dimension.
    pub counts: Array1<Ix>,
}

impl Cells {
    pub fn new(opencl: &OpenCLCtx, counts: &Array1<Ix>, q: usize) -> Self {
        let n = counts.product() as usize;
        let d = counts.dim();

        // Allocate the buffers.
        let mut out = Self {
            typ: Data::new(opencl, [n], 0),
            idx: Data::new(opencl, [n, q], 0),
            counts: counts.clone(),
        };

        // Populate the upstream indices
        out.idx.host = VelocitySet::make(d, q).upstream_idx(counts);

        out.write_to_dev(opencl);

        out
    }

    // Copy data from our host arrays into the OpenCL buffers.
    pub fn write_to_dev(&mut self, opencl: &OpenCLCtx) {
        let queue = &opencl.queue;
        let _write_t = self.typ.enqueue_write(queue, "typ");
        let _write_i = self.idx.enqueue_write(queue, "idx");

        queue
            .finish()
            .expect("queue.finish() failed [Cells::write_to_dev]");
    }
}

impl MemUsage for Cells {
    fn size_bytes(&self) -> usize {
        std::mem::size_of::<i32>() * (self.typ.host.len() + self.idx.host.len())
    }
}

/// See [`FluidDeserializer`]
#[derive(Deserialize)]
struct CellsDeserializer {
    typ: Data1dDeserializer<i32>,
    idx: Data2dDeserializer<i32>,
    counts: Array1<Ix>,
}

impl CtxDeserializer for CellsDeserializer {
    type Target = Cells;

    fn with_context(self, opencl: &OpenCLCtx) -> Self::Target {
        Self::Target {
            typ: self.typ.with_context(opencl),
            idx: self.idx.with_context(opencl),
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
    pub fn new(opencl: OpenCLCtx, counts: Array1<impl IxLike>, q: usize, omega_ns: f32) -> Self {
        let counts = counts.mapv(|c| c.into());
        let cells = Cells::new(&opencl, &counts, q);
        let fluid = Fluid::new(&opencl, &counts, q, omega_ns);

        Self {
            opencl: opencl,
            cells: cells,
            fluid: fluid,
            tracers: Vec::new(),
            gravity: Array1::zeros([counts.dim()]),
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

        let cell_count = self.cells.counts.product() as usize;

        // TODO: make this generic over [`D`], [`Q`] and [`EqnType`]!
        let d2q9 = &self.opencl.d2q9_ns_kernel;
        let d2q5 = &self.opencl.d2q5_ad_kernel;

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
                    .set_global_work_size(cell_count)
                    .enqueue_nd_range(&queue)
                    .expect("ExecuteKernel::new failed.")
            };
        }

        for iter in 0..iters {
            // We cannot pass bools to OpenCL so get an interger with values 0 or 1.
            let even = (1 - iter % 2) as i32;

            // let d2q9_event = unsafe {
            //     exec_kernel!(
            //         d2q9,
            //         &even,
            //         &self.omega_ns,
            //         &self.gravity[0],
            //         &self.gravity[1],
            //         &mut self.fluid.f.dev,
            //         &mut self.fluid.rho.dev,
            //         &mut self.fluid.vel.dev,
            //         &self.cells.typ.dev,
            //         &self.cells.idx.dev
            //     )
            // };

            let d2q9_event = unsafe {
                ExecuteKernel::new(&d2q9)
                    .set_arg(&even)
                    .set_arg(&self.fluid.omega)
                    .set_arg(&self.gravity[0])
                    .set_arg(&self.gravity[1])
                    .set_arg(&mut self.fluid.f.dev)
                    .set_arg(&mut self.fluid.rho.dev)
                    .set_arg(&mut self.fluid.vel.dev)
                    .set_arg(&self.cells.typ.dev)
                    .set_arg(&self.cells.idx.dev)
                    .set_local_work_size(64)
                    .set_global_work_size(cell_count)
                    .enqueue_nd_range(&queue)
                    .expect("ExecuteKernel::new failed.")
            };

            events.push(d2q9_event.get());
            queue.finish().expect("queue.finish failed");

            for tracer in self.tracers.iter_mut() {
                let d2q5_event = unsafe {
                    ExecuteKernel::new(&d2q5)
                        .set_arg(&even)
                        .set_arg(&tracer.omega)
                        .set_arg(&mut tracer.g.dev)
                        .set_arg(&mut tracer.C.dev)
                        .set_arg(&self.fluid.vel.dev)
                        .set_arg(&self.cells.typ.dev)
                        .set_arg(&self.cells.idx.dev)
                        .set_global_work_size(cell_count)
                        .enqueue_nd_range(&queue)
                        .expect("ExecuteKernel::new failed.")
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
