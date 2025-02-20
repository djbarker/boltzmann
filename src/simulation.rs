use ndarray::Array1;
use opencl3::command_queue::CommandQueue;
use opencl3::kernel::ExecuteKernel;
use opencl3::types::cl_event;

use crate::fields::{Fluid, MemUsage, Scalar};
use crate::raster::IxLike;
use crate::{
    opencl::{Data, Data1d, Data2d, OpenCLCtx},
    raster::Ix,
    velocities::VelocitySet,
};

/// Contains information about the cells.
/// For example, whether they are a fluid or wall, have fixed velocity, etc.
pub struct Cells {
    /// The type of the cell as given by [`CellType`].
    pub typ: Data1d<i32>,

    /// The _upstream_ cell indices in each direction.
    /// NOTE: Not public because it's basically an implementation detail.
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

#[repr(C)]
pub enum CellType {
    FLUID = 0,
    WALL = 1,
    FIXED = 2,
}

/// Stores the [`Fluid`] and any associated [`Scalar`] fields, calls the OpenCL kernels.
pub struct Simulation {
    pub opencl: OpenCLCtx,
    pub cells: Cells,
    pub fluid: Fluid,
    pub tracers: Vec<Scalar>,
    pub gravity: Array1<f32>,
}

impl Simulation {
    pub fn new(counts: Array1<impl IxLike>, q: usize, omega_ns: f32) -> Self {
        let counts = counts.mapv(|c| c.into());
        let opencl = OpenCLCtx::new();
        let cells = Cells::new(&opencl, &counts, q);
        let fluid = Fluid::new(&opencl, &counts, q, omega_ns);

        Self {
            opencl: opencl,
            cells: cells,
            fluid: fluid,
            tracers: Vec::new(),
            gravity: Array1::zeros([counts.dim()]),
        }
    }

    pub fn set_gravity(&mut self, gravity: Array1<f32>) {
        self.gravity = gravity;
    }

    pub fn add_tracer(&mut self, tracer: Scalar) {
        self.tracers.push(tracer);
    }

    /// Copy data to OpenCL buffers & optionally calculate distribution function from macroscopic variables.
    ///
    /// Once we've initialized the density, velocity, etc arrays in Python we must call [`Simulation::finalize`]
    /// before calling [`Simulation::iterate`] so that the data is on the OpenCL device.
    /// If we are not loading from a checkpoint we also first need to initialize the distribution
    /// functions by calculating the equilibrium distributions.
    pub fn finalize(&mut self, equilibrate: bool) {
        self.cells.write_to_dev(&self.opencl);

        if equilibrate {
            self.fluid.equilibrate();
            for tracer in self.tracers.iter_mut() {
                tracer.equilibrate(self.fluid.vel.host.view());
            }
        }

        self.fluid.write_to_dev(&self.opencl);
        for tracer in self.tracers.iter_mut() {
            tracer.write_to_dev(&self.opencl);
        }
    }

    /// See: opencl3 example [`basic.rs`](https://github.com/kenba/opencl3/blob/main/examples/basic.rs)
    pub fn iterate(&mut self, iters: usize) {
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
    }
}
