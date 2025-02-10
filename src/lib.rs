use std::iter::zip;
use std::ptr;
use std::sync::{Arc, Mutex, MutexGuard};

use ndarray::{
    arr1, arr2, Array, Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, Axis, Dimension,
    ShapeBuilder, Zip,
};
use numpy::{Ix1, Ix2, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1};
use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
use opencl3::context::Context;
use opencl3::device::{
    get_all_devices, Device, CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_DEFAULT, CL_DEVICE_TYPE_GPU,
};
use opencl3::event::Event;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_WRITE};
use opencl3::program::Program;
use opencl3::types::{cl_event, CL_BLOCKING};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use raster::StrideOrder::RowMajor;
use raster::{counts_to_strides, idx_to_sub, raster_row_major, sub_to_idx, Ix, IxLike};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use utils::vmod_nd;

pub mod raster;
pub mod utils;
pub mod vect_d;
pub mod vect_s;

/// Contains the weights & velocities for calculating offsets and equilibrium distribution functions.
/// NOTE: The order matches that used in [`lib.cl`].
#[derive(Clone)]
pub struct VelocitySet {
    ws: Array1<f32>,
    qs: Array2<i32>,
}

macro_rules! ax0 {
    ($arr:expr, $idx:expr) => {
        ($arr).index_axis(Axis(0), $idx)
    };
}

/// Returns the outer product of the two given vectors.
/// The vectors need not have the same shape but in our use-case they always do.
fn outer(x: ArrayView1<f32>, y: ArrayView1<f32>) -> Array2<f32> {
    let mut z = Array2::zeros([x.dim(), y.dim()]);
    for i in 0..x.dim() {
        for j in 0..y.dim() {
            z[(i, j)] = x[i] * y[j]
        }
    }
    z
}

/// Returns the double dot product of two matrices.
/// The matrices must have the same shape.
fn doubledot(x: ArrayView2<f32>, y: ArrayView2<f32>) -> f32 {
    assert_eq!(x.dim(), y.dim());
    let mut z = 0.0;
    for i in 0..x.dim().0 {
        for j in 0..x.dim().1 {
            z += x[(i, j)] * y[(i, j)]
        }
    }
    z
}

/// Returns the n-dimensional identity matrix.
fn ident(n: usize) -> Array2<f32> {
    Array2::from_diag_elem(n, 1.0)
}

#[allow(non_snake_case)]
impl VelocitySet {
    /// The dimension of the [`VelocitySet`].
    pub fn D(&self) -> usize {
        self.qs.dim().1
    }

    /// The velocity set size of the [`VelocitySet`].
    pub fn Q(&self) -> usize {
        self.qs.dim().0
    }

    /// Calculate the equilibrium distribution function for a single cell.
    /// The returned array has shape == [Q]
    ///
    /// TODO: I seem to have needed a lot of clone/view calls here. I suspect this could be cleaner.
    fn feq(&self, rho: f32, vel: ArrayView1<f32>) -> Array1<f32> {
        let I = ident(self.D());
        let mut out: Array1<f32> = Array1::zeros([self.Q()]);
        for q in 0..self.Q() {
            let q_ = ax0!(self.qs, q).mapv(|x| x as f32);
            let vq = vel.dot(&q_);
            let vv = vel.dot(&vel);
            out[q] = rho * self.ws[q] * (1.0 + 3.0 * vq + 4.5 * vq * vq - (3.0 / 2.0) * vv)
        }
        out
    }

    fn make(d: usize, q: usize) -> VelocitySet {
        match (d, q) {
            (2, 9) => Self::D2Q9(),
            (2, 5) => Self::D2Q5(),
            _ => panic!("Unknown model: D{}Q{}", d, q),
        }
    }

    fn D2Q9() -> VelocitySet {
        VelocitySet {
            ws: arr1(&[
                4. / 9.,
                1. / 9.,
                1. / 9.,
                1. / 9.,
                1. / 9.,
                1. / 36.,
                1. / 36.,
                1. / 36.,
                1. / 36.,
            ]),
            qs: arr2(&[
                [0, 0],
                [1, 0],
                [-1, 0],
                [0, 1],
                [0, -1],
                [1, 1],
                [-1, -1],
                [-1, 1],
                [1, -1],
            ]),
        }
    }

    fn D2Q5() -> VelocitySet {
        VelocitySet {
            ws: arr1(&[1. / 3., 1. / 6., 1. / 6., 1. / 6., 1. / 6.]),
            qs: arr2(&[[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]]),
        }
    }
}

/// The implemented OpenCL kernels differ in whether they update the velocity or not.
/// [`EqnType::NavierStokes`] does, whereas [`EqnType::AdvectionDiffusion`] does not.
pub enum EqnType {
    NavierStokes,
    AdvectionDiffusion,
}

/// Return an array of size (cell_count, q) where for each cell we have calculated the index of the
/// cell upstream of each velocity in the model.
pub fn upstream_idx(counts: &Array1<Ix>, model: VelocitySet) -> Array2<i32> {
    let strides = counts_to_strides(counts, RowMajor);
    let mut idx = Array2::zeros([cell_count(counts), model.Q()]);
    let mut i = 0;
    for sub in raster_row_major(counts.clone()) {
        for q in 0..model.Q() {
            let sub_ = sub.clone() + ax0!(model.qs, q).mapv(|q| q as Ix);
            let sub_ = vmod_nd(sub_, &counts);
            let j = sub_to_idx(&sub_, &strides);
            idx[(i, q)] = j as Ix;
        }

        i += 1;
    }

    return idx;
}

/// Calculate the 2D curl (i.e. the z component of the 3D curl).
/// All quantities are in lattice units.
/// TODO: openCL version
/// TODO: would be nice to be able to _return_ the array to python
fn calc_curl_2d(
    vel: &ArrayView2<f32>,
    cells: &ArrayView1<i32>,
    counts: &ArrayView1<impl Into<Ix> + Copy>,
    curl: &mut ArrayViewMut1<f32>,
) {
    assert_eq!(vel.dim().0, cells.dim());

    fn is_wall(c: i32) -> f32 {
        ((c == (CellType::WALL as i32)) as i32) as f32
    }

    let counts = counts.to_owned().mapv(|x| x.into());
    let strides = &counts_to_strides(&counts, RowMajor);

    let offset = |sub: &Array1<Ix>, off: [Ix; 2]| {
        let off = arr1(&off);
        sub_to_idx(&vmod_nd(sub + off, &counts), &strides)
    };

    curl.as_slice_mut()
        .unwrap()
        .par_iter_mut()
        .enumerate()
        .for_each(|(idx, c)| {
            let sub = idx_to_sub(idx, &strides, RowMajor);

            let idx_x1 = offset(&sub, [-1, 0]);
            let idx_x2 = offset(&sub, [1, 0]);
            let idx_y1 = offset(&sub, [0, -1]);
            let idx_y2 = offset(&sub, [0, 1]);

            // NOTE: Assumes zero wall velocity.
            let dvydx1: f32 = vel[(idx_x1, 1)] * (1.0 - is_wall(cells[idx_x1]));
            let dvydx2: f32 = vel[(idx_x2, 1)] * (1.0 - is_wall(cells[idx_x2]));
            let dvxdy1: f32 = vel[(idx_y1, 0)] * (1.0 - is_wall(cells[idx_y1]));
            let dvxdy2: f32 = vel[(idx_y2, 0)] * (1.0 - is_wall(cells[idx_y2]));

            *c = ((dvydx2 - dvydx1) - (dvxdy2 - dvxdy1)) / 2.0;
        });

    // for sub in raster_row_major(counts.clone()) {
    //     let idx = sub_to_idx(&sub, strides);

    //     let idx_x1 = offset(&sub, [-1, 0]);
    //     let idx_x2 = offset(&sub, [1, 0]);
    //     let idx_y1 = offset(&sub, [0, -1]);
    //     let idx_y2 = offset(&sub, [0, 1]);

    //     // NOTE: Assumes zero wall velocity.
    //     let dvydx1: f32 = vel[(idx_x1, 1)] * (1.0 - is_wall(cells[idx_x1]));
    //     let dvydx2: f32 = vel[(idx_x2, 1)] * (1.0 - is_wall(cells[idx_x2]));
    //     let dvxdy1: f32 = vel[(idx_y1, 0)] * (1.0 - is_wall(cells[idx_y1]));
    //     let dvxdy2: f32 = vel[(idx_y2, 0)] * (1.0 - is_wall(cells[idx_y2]));

    //     curl[idx] = ((dvydx2 - dvydx1) - (dvxdy2 - dvxdy1)) / 2.0;
    // }
}

#[repr(C)]
pub enum CellType {
    FLUID = 0,
    WALL = 1,
    FIXED = 2,
}

const OPENCL_SRC: &str = include_str!("lib.cl");

/// Long-lived OpenCL context.
/// Exists so we do not need to recreate everthing for each batch of iterations.
struct OpenCLCtx {
    _device: Device,
    context: Context,
    queue: CommandQueue,
    d2q9_ns_kernel: Kernel,
    d2q5_ad_kernel: Kernel,
}

impl OpenCLCtx {
    fn new() -> Self {
        // GPU: CL_DEVICE_TYPE_GPU, CPU: CL_DEVICE_TYPE_CPU
        let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
            .expect("error getting platform")
            .first()
            .expect("no device found in platform");
        let device = Device::new(device_id);
        let context = Context::from_device(&device).expect("Context::from_device failed");
        let queue = CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE)
            .expect("CommandQueue::create_default failed");

        let program = Program::create_and_build_from_source(&context, OPENCL_SRC, "")
            .expect("Program::create_and_build_from_source failed");
        let d2q9 = Kernel::create(&program, "update_d2q9_bgk").expect("Kernel::create failed D2Q9");
        let d2q5 = Kernel::create(&program, "update_d2q5_bgk").expect("Kernel::create failed D2Q5");

        Self {
            _device: device,
            context: context,
            queue: queue,
            d2q9_ns_kernel: d2q9,
            d2q5_ad_kernel: d2q5,
        }
    }
}

/// Long-lived container for host- and OpenCL device buffers.
struct Data<T, D>
where
    D: Dimension,
{
    pub host: Array<T, D>,
    pub dev: Buffer<T>,
}

impl<T, D: Dimension> Data<T, D> {
    /// Consume an [`Array`] and turn it into a [`Data`].
    pub fn new<Sh>(opencl: &OpenCLCtx, shape: Sh, fill: T) -> Self
    where
        T: Clone,
        Sh: ShapeBuilder<Dim = D>,
    {
        let host = Array::from_elem(shape, fill);
        Self {
            dev: Data::make_buffer(&opencl.context, &host),
            host,
        }
    }

    /// Construct an OpenCL [`Buffer`] from the passed [`VectDView`].
    fn make_buffer(ctx: &Context, arr_host: &Array<T, D>) -> Buffer<T> {
        unsafe {
            Buffer::create(
                ctx,
                CL_MEM_READ_WRITE,
                arr_host.raw_dim().size(),
                ptr::null_mut(),
            )
            .expect("Buffer::create() failed")
        }
    }

    /// Write our host array to the OpenCL [`Buffer`].
    fn enqueue_write(&mut self, queue: &CommandQueue, msg: &str) -> Event {
        unsafe {
            queue
                .enqueue_write_buffer(
                    &mut self.dev,
                    CL_BLOCKING,
                    0,
                    self.host.as_slice().unwrap(),
                    &[], // ignore events ... why?
                )
                .expect(format!("enqueue_write_buffer failed {}", msg).as_str())
        }
    }

    // Read back from OpenCL device buffers into our host arrays.
    fn enqueue_read(&mut self, queue: &CommandQueue) -> Event {
        assert_eq!(self.host.len(), self.host.as_slice().unwrap().len());
        unsafe {
            queue
                .enqueue_read_buffer(
                    &self.dev,
                    CL_BLOCKING,
                    0,
                    self.host.as_slice_mut().unwrap(),
                    &[], // ignore events ... why?
                )
                .expect("enqueue_read_buffer failed")
        }
    }
}

type Data1d<T> = Data<T, Ix1>;
type Data2d<T> = Data<T, Ix2>;

trait MemUsage {
    fn size_bytes(&self) -> usize;
}

/// Long-lived container for the fluid Host and OpenCL device buffers.
/// This owns the simulation data and we expose a view to Python.
struct Fluid {
    f: Data2d<f32>,
    rho: Data1d<f32>,
    vel: Data2d<f32>,
    model: VelocitySet,
}

impl Fluid {
    pub fn new(opencl: &OpenCLCtx, counts: &Array1<Ix>, q: usize) -> Self {
        let n = cell_count(counts);
        let d = counts.dim();

        Self {
            f: Data::new(opencl, [n, q], 0.0),
            rho: Data::new(opencl, [n], 1.0),
            vel: Data::new(opencl, [n, d], 0.0),
            model: VelocitySet::make(d, q),
        }
    }

    // Copy data from our host arrays into the OpenCL buffers.
    pub fn write_to_dev(&mut self, opencl: &OpenCLCtx) {
        let queue = &opencl.queue;
        let _write_f = self.f.enqueue_write(queue, "f");
        let _write_r = self.rho.enqueue_write(queue, "rho");
        let _write_v = self.vel.enqueue_write(queue, "vel");

        queue
            .finish()
            .expect("queue.finish() failed [Fluid::write_to_dev]");
    }

    /// Read the data back from the OpenCL buffers into our host arrays.
    pub fn read_to_host(&mut self, opencl: &OpenCLCtx) {
        let queue = &opencl.queue;
        let _read_f = self.f.enqueue_read(queue);
        let _read_r = self.rho.enqueue_read(queue);
        let _read_v = self.vel.enqueue_read(queue);

        queue
            .finish()
            .expect("queue.finish() failed [Fluid::read_to_host]");
    }

    pub fn equilibrate(&mut self) {
        Zip::from(&self.rho.host)
            .and(self.vel.host.rows())
            .and(self.f.host.rows_mut())
            .for_each(|&r, v, mut f| {
                f.assign(&self.model.feq(r, v));
            });
    }
}

impl MemUsage for Fluid {
    fn size_bytes(&self) -> usize {
        std::mem::size_of::<f32>() * (self.f.host.len() + self.rho.host.len() + self.vel.host.len())
    }
}

/// Long-lived container for scalar field Host and OpenCL device buffers.
/// This owns the simulation data and we expose a view to Python.
#[allow(non_snake_case)]
struct Scalar {
    g: Data2d<f32>,
    C: Data1d<f32>,
    model: VelocitySet,
}

#[allow(non_snake_case)]
impl Scalar {
    pub fn new(opencl: &OpenCLCtx, counts: &Array1<i32>, q: usize) -> Self {
        let n = cell_count(counts);
        let d = counts.dim();

        Self {
            g: Data::new(opencl, [n, q], 0.0),
            C: Data::new(opencl, [n], 0.0),
            model: VelocitySet::make(d, q),
        }
    }

    // Copy data from our host arrays into the OpenCL buffers.
    pub fn write_to_dev(&mut self, opencl: &OpenCLCtx) {
        let queue = &opencl.queue;
        let _write_g = self.g.enqueue_write(queue, "q");
        let _write_C = self.C.enqueue_write(queue, "C");

        queue
            .finish()
            .expect("queue.finish() failed [Scalar::write_to_dev]");
    }

    /// Read the data back from the OpenCL buffers into our host arrays.
    pub fn read_to_host(&mut self, opencl: &OpenCLCtx) {
        let queue = &opencl.queue;
        let _read_g = self.g.enqueue_read(queue);
        let _read_C = self.C.enqueue_read(queue);

        queue
            .finish()
            .expect("queue.finish() failed [Scalar::read_to_host]");
    }

    pub fn equilibrate(&mut self, vel: ArrayView2<f32>) {
        Zip::from(&self.C.host)
            .and(vel.rows())
            .and(self.g.host.rows_mut())
            .for_each(|&C, v, mut g| {
                g.assign(&self.model.feq(C, v));
            });
    }
}

impl MemUsage for Scalar {
    fn size_bytes(&self) -> usize {
        std::mem::size_of::<f32>() * (self.g.host.len() + self.C.host.len())
    }
}

struct Cells {
    /// The type of the cell as given by [`CellType`].
    typ: Data1d<i32>,

    /// The _upstream_ cell indices in each direction.
    idx: Data2d<i32>,

    /// Cell count in each dimension.
    counts: Array1<Ix>,
}

/// Return the total number of cells for the given dimension sizes.
fn cell_count(counts: &Array1<Ix>) -> usize {
    counts.product() as usize
}

impl Cells {
    pub fn new(opencl: &OpenCLCtx, counts: &Array1<Ix>, q: usize) -> Self {
        let n = cell_count(counts);
        let d = counts.dim();

        // Allocate the buffers.
        let mut out = Self {
            typ: Data::new(opencl, [n], 0),
            idx: Data::new(opencl, [n, q], 0),
            counts: counts.clone(),
        };

        // Populate the upstream indices
        out.idx.host = upstream_idx(counts, VelocitySet::make(d, q));

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

struct Simulation {
    opencl: OpenCLCtx,
    cells: Cells,
    fluid: Fluid,
    tracer: Option<Scalar>,
    omega_ns: f32,
    omega_ad: f32,
    gravity: Array1<f32>,
}

impl Simulation {
    pub fn new(counts: Array1<impl IxLike>, q: usize, omega_ns: f32) -> Self {
        let counts = counts.mapv(|c| c.into());
        let opencl = OpenCLCtx::new();
        let cells = Cells::new(&opencl, &counts, q);
        let fluid = Fluid::new(&opencl, &counts, q);

        Self {
            opencl: opencl,
            cells: cells,
            fluid: fluid,
            tracer: None,
            omega_ns: omega_ns,
            omega_ad: f32::NAN,
            gravity: Array1::zeros([counts.dim()]),
        }
    }

    pub fn set_gravity(&mut self, gravity: Array1<f32>) {
        self.gravity = gravity;
    }

    pub fn add_tracer(&mut self, tracer: Scalar, omega_ad: f32) {
        self.tracer = Some(tracer);
        self.omega_ad = omega_ad;
    }

    /// Calculate distribution function from macroscopic variables & copy data to OpenCL buffers.
    pub fn finalize(&mut self, equilibrate: bool) {
        self.cells.write_to_dev(&self.opencl);

        if equilibrate {
            self.fluid.equilibrate();
        }
        self.fluid.write_to_dev(&self.opencl);

        if let Some(ref mut tracer) = self.tracer {
            if equilibrate {
                tracer.equilibrate(self.fluid.vel.host.view());
            }
            tracer.write_to_dev(&self.opencl);
        }
    }

    /// See: opencl3 example [`basic.rs`](https://github.com/kenba/opencl3/blob/main/examples/basic.rs)
    pub fn iterate(&mut self, iters: usize) {
        if iters % 2 != 0 {
            panic!("iters must be even")
        }

        let queue: &CommandQueue = &self.opencl.queue;

        let cell_count = cell_count(&self.cells.counts);

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
                    .set_arg(&self.omega_ns)
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

            // We do this if let binding on every iteration.
            // I assume it's cheap but I wonder if that's really so.
            // One could imagine we make the iteration generic and it gets a list of funcs to call.
            // The borrow checker may kill us here though. :(
            if let Some(ref mut tracer) = &mut self.tracer {
                let d2q5_event = unsafe {
                    ExecuteKernel::new(&d2q5)
                        .set_arg(&even)
                        .set_arg(&self.omega_ad)
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
        if let Some(ref mut tracer) = self.tracer {
            tracer.read_to_host(&self.opencl);
        }
    }
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
}

#[pymethods]
impl ScalarPy {
    #[getter]
    fn g<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray2<f32>> {
        let borrow = this.borrow();
        let sim = borrow.sim.lock().unwrap();
        if let Some(ref tracer) = sim.tracer {
            let array = &tracer.g.host;
            unsafe { PyArray2::borrow_from_array(array, this.into_any()) }
        } else {
            // Just panic here because we shouldn't be able to get the [`ScalarPy`]
            // object if the [`Simulation`] doesn't have a tracer configured.
            panic!("No tracer configured")
        }
    }

    #[getter]
    fn val<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray1<f32>> {
        let borrow = this.borrow();
        let sim = borrow.sim.lock().unwrap();
        if let Some(ref tracer) = sim.tracer {
            let array = &tracer.C.host;
            unsafe { PyArray1::borrow_from_array(array, this.into_any()) }
        } else {
            // Just panic here because we shouldn't be able to get the [`ScalarPy`]
            // object if the [`Simulation`] doesn't have a tracer configured.
            panic!("No tracer configured")
        }
    }

    #[getter]
    fn size_bytes(&self) -> usize {
        self.sim
            .lock()
            .unwrap()
            .tracer
            .as_ref()
            .unwrap()
            .size_bytes()
    }
}

#[pyclass(name = "Domain")]
struct DomainPy {
    sim: Arc<Mutex<Simulation>>,
}

#[pymethods]
impl DomainPy {
    #[getter]
    fn cell_type<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray1<i32>> {
        let borrow = this.borrow();
        let sim = borrow.sim.lock().unwrap();
        let array = &sim.cells.typ.host;
        unsafe { PyArray1::borrow_from_array(array, this.into_any()) }
    }

    #[getter]
    fn offset_idx<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray2<i32>> {
        let borrow = this.borrow();
        let sim = borrow.sim.lock().unwrap();
        let array = &sim.cells.idx.host;
        unsafe { PyArray2::borrow_from_array(array, this.into_any()) }
    }

    #[getter]
    fn size_bytes(&self) -> usize {
        self.sim.lock().unwrap().cells.size_bytes()
    }
}

/// Wrap the core [`Simulation`] class in an [`Arc`] + [`Mutex`] so it's threadsafe for Python.
#[pyclass(name = "Simulation")]
struct SimulationPy {
    sim: Arc<Mutex<Simulation>>,
}

impl SimulationPy {
    fn sim(&mut self) -> MutexGuard<'_, Simulation> {
        self.sim.lock().expect("Aquiring simulation mutex failed.")
    }
}

#[pymethods]
impl SimulationPy {
    #[new]
    fn new(counts: PyReadonlyArray1<i32>, q: usize, omega_ns: f32) -> Self {
        let counts = counts.as_array().to_owned();
        Self {
            sim: Arc::new(Mutex::new(Simulation::new(counts, q, omega_ns))),
        }
    }

    /// Once we've initialized the density, velocity, etc arrays in Python we must call [`SimulationPy::finalize`] before calling [`SimulationPy::iterate`].
    /// This calculates the equilibrium distribution function and copies the necessary data to the OpenCL device buffers.
    fn finalize(&mut self, equilibrate: bool) {
        self.sim().finalize(equilibrate);
    }

    fn iterate(&mut self, iters: usize) {
        self.sim().iterate(iters)
    }

    fn set_gravity(&mut self, gravity: PyReadonlyArray1<f32>) {
        let gravity = gravity.as_array().to_owned();
        self.sim().set_gravity(gravity);
    }

    fn add_tracer(&mut self, q: usize, omega_ad: f32) {
        let mut sim = self.sim();
        let c = &sim.cells.counts;
        let tracer = Scalar::new(&sim.opencl, c, q);
        sim.add_tracer(tracer, omega_ad)
    }

    #[getter]
    fn domain<'py>(this: Bound<'py, Self>) -> Bound<'py, DomainPy> {
        let this = this.borrow();
        Bound::new(
            this.py(),
            DomainPy {
                sim: this.sim.clone(),
            },
        )
        .expect("Bound::new DomainPy failed.")
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

    #[getter]
    fn tracer<'py>(this: Bound<'py, Self>) -> PyResult<Bound<'py, ScalarPy>> {
        let this = this.borrow();
        let sim = this.sim.lock().unwrap();
        if sim.tracer.is_some() {
            let bound = Bound::new(
                this.py(),
                ScalarPy {
                    sim: this.sim.clone(),
                },
            )
            .expect("Bound::new FluidPy failed.");
            Ok(bound)
        } else {
            Err(PyValueError::new_err("Simulation has no tracer configured"))
        }
    }
}

/// Computationally heavy parts of our LBM simulation, written in Rust.
///
///
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
    ) {
        calc_curl_2d(
            &vel.as_array(),
            &cells.as_array(),
            &counts.as_array(),
            &mut curl.as_array_mut(),
        );
    }

    m.add_class::<SimulationPy>()?;
    m.add_class::<FluidPy>()?;
    m.add_class::<ScalarPy>()?;
    m.add_class::<DomainPy>()?;

    Ok(())
}
