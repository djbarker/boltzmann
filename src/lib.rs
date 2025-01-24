use std::sync::{Arc, Mutex};
use std::{ptr, str, usize};

use numpy::{PyReadwriteArray1, PyReadwriteArray2};
use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_DEFAULT};
use opencl3::event::Event;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_WRITE};
use opencl3::program::Program;
use opencl3::types::{cl_event, CL_BLOCKING};
use pyo3::prelude::*;
use raster::{counts_to_strides, sub_to_idx, Counts, Raster, Sub};
use utils::vmod;
use vect_d::{ArrayD, Data, VectD, VectDView, VectDViewScalar, VectDViewVector};
use vect_s::VectS;

// mod lbm;
pub mod domain;
pub mod raster;
pub mod utils;
pub mod vect_d;
pub mod vect_s;

/// PONDER: Making this generic over D & Q gives us some nice compile-time checks, but it makes
///         exposing to Python much more clunky. Is it worth doing away with static D & Q and making
///         them runtime? I wonder if there are also any perf benefits from knowing them at
///         compile-time?
pub struct Model<const D: usize, const Q: usize> {
    w: [f32; Q],
    q: [VectS<i32, D>; Q],
}

impl<const D: usize, const Q: usize> Model<D, Q> {
    const D: usize = D;
    const Q: usize = Q;
}

pub static D2Q9: Model<2, 9> = Model::<2, 9> {
    w: [
        4. / 9.,
        1. / 9.,
        1. / 9.,
        1. / 9.,
        1. / 9.,
        1. / 36.,
        1. / 36.,
        1. / 36.,
        1. / 36.,
    ],
    q: [
        VectS::new([0, 0]),
        VectS::new([1, 0]),
        VectS::new([-1, 0]),
        VectS::new([0, 1]),
        VectS::new([0, -1]),
        VectS::new([1, 1]),
        VectS::new([-1, -1]),
        VectS::new([-1, 1]),
        VectS::new([1, -1]),
    ],
};

/// Calculate the 2D curl (i.e. the z component of the 3D curl).
/// All quantities are in lattice units.
/// TODO: openCL version
/// TODO: would be nice to be able to _return_ the array to python
fn calc_curl_2d(
    vel: &VectDViewVector<f32, 2>,
    cells: &VectDViewScalar<i32>,
    indices: &VectDViewVector<i32, 9>,
    curl: &mut VectDViewScalar<f32>,
) {
    assert_eq!(vel.data_count(), indices.data_count());

    fn btof32(b: bool) -> f32 {
        (b as i32) as f32
    }

    for i in 0..vel.data_count() {
        let idx = indices[i];
        // NOTE: Assumes zero wall velocity.
        let dvydx1 = vel[idx[2]][1] * btof32(cells[idx[2]] != (CellType::WALL as i32));
        let dvydx2 = vel[idx[1]][1] * btof32(cells[idx[1]] != (CellType::WALL as i32));
        let dvxdy1 = vel[idx[4]][0] * btof32(cells[idx[4]] != (CellType::WALL as i32));
        let dvxdy2 = vel[idx[3]][0] * btof32(cells[idx[3]] != (CellType::WALL as i32));

        curl[i] = ((dvydx2 - dvydx1) - (dvxdy2 - dvxdy1)) / 2.0;
    }
}

fn update_d2q5_fixed(
    even: bool,
    f: &mut VectDViewVector<f32, 5>,
    val: &VectDViewScalar<f32>,
    vel: &VectDViewVector<f32, 2>,
    idx: VectS<i32, 9>, // We can re-use the D2Q9 indices because the first 5 velocities match.
) {
    let val = val[idx[0]];
    let vel = vel[idx[0]];
    let vx = vel[0];
    let vy = vel[1];
    let vv = vx * vx + vy * vy;
    let vxx = vx * vx;
    let vyy = vy * vy;

    let feq = [
        val * (1.0 / 6.0) * (2.0 - 3.0 * vv),
        val * (1.0 / 12.0) * (2.0 + 6.0 * vx + 9.0 * vxx - 3.0 * vv),
        val * (1.0 / 12.0) * (2.0 - 6.0 * vx + 9.0 * vxx - 3.0 * vv),
        val * (1.0 / 12.0) * (2.0 + 6.0 * vy + 9.0 * vyy - 3.0 * vv),
        val * (1.0 / 12.0) * (2.0 - 6.0 * vy + 9.0 * vyy - 3.0 * vv),
    ];

    // write back to same locations
    if even {
        f[idx[0]][0] = feq[0];
        f[idx[1]][1] = feq[2];
        f[idx[2]][2] = feq[1];
        f[idx[3]][3] = feq[4];
        f[idx[4]][4] = feq[3];
    } else {
        for i in 0..5 {
            f[idx[0]][i] = feq[i];
        }
    }
}

fn update_d2q9_fixed(
    even: bool,
    f: &mut impl ArrayD<VectS<f32, 9>>,
    rho: &impl ArrayD<f32>,
    vel: &impl ArrayD<VectS<f32, 2>>,
    idx: VectS<i32, 9>,
) {
    let r = rho[idx[0]];
    let vel = vel[idx[0]];
    let vx = vel[0];
    let vy = vel[1];
    let vv = vx * vx + vy * vy;
    let vxx = vx * vx;
    let vyy = vy * vy;
    let vxy = vx * vy;

    let feq = [
        r * (2.0 / 9.0) * (2.0 - 3.0 * vv),
        r * (1.0 / 18.0) * (2.0 + 6.0 * vx + 9.0 * vxx - 3.0 * vv),
        r * (1.0 / 18.0) * (2.0 - 6.0 * vx + 9.0 * vxx - 3.0 * vv),
        r * (1.0 / 18.0) * (2.0 + 6.0 * vy + 9.0 * vyy - 3.0 * vv),
        r * (1.0 / 18.0) * (2.0 - 6.0 * vy + 9.0 * vyy - 3.0 * vv),
        r * (1.0 / 36.0) * (1.0 + 3.0 * (vx + vy) + 9.0 * vxy + 3.0 * vv),
        r * (1.0 / 36.0) * (1.0 - 3.0 * (vx + vy) + 9.0 * vxy + 3.0 * vv),
        r * (1.0 / 36.0) * (1.0 + 3.0 * (vy - vx) - 9.0 * vxy + 3.0 * vv),
        r * (1.0 / 36.0) * (1.0 - 3.0 * (vy - vx) - 9.0 * vxy + 3.0 * vv),
    ];

    // write back to same locations
    if even {
        f[idx[0]][0] = feq[0];
        f[idx[1]][1] = feq[2];
        f[idx[2]][2] = feq[1];
        f[idx[3]][3] = feq[4];
        f[idx[4]][4] = feq[3];
        f[idx[5]][5] = feq[6];
        f[idx[6]][6] = feq[5];
        f[idx[7]][7] = feq[8];
        f[idx[8]][8] = feq[7];
    } else {
        for i in 0..9 {
            f[idx[0]][i] = feq[i];
        }
    }
}

#[rustfmt::skip]
fn update_d2q5_bgk(
    even: bool,
    omega: f32,
    f: &mut impl ArrayD<VectS<f32, 5>>,
    val: &mut impl ArrayD<f32>,
    vel: &impl ArrayD<VectS<f32, 2>>,
    idx: VectS<i32, 9>,  // We can re-use the D2Q9 indices because the first 5 velocities match.
) {
    // 0:  0  0
    // 1: +1  0
    // 2: -1  0
    // 3:  0 +1
    // 4:  0 -1

    // collect fs
    let mut f_ = VectS::new(if even {
        [
            f[idx[0]][0],
            f[idx[1]][1],
            f[idx[2]][2],
            f[idx[3]][3],
            f[idx[4]][4],
        ]
    } else {
        [
            f[idx[0]][0],
            f[idx[0]][2],
            f[idx[0]][1],
            f[idx[0]][4],
            f[idx[0]][3],
        ]
    });

    // calc moments
    let r = f_.sum();
    let v = vel[idx[0]];
    let vv = (v * v).sum();
    let vxx = v[0] * v[0];
    let vyy = v[1] * v[1];
    
    // calc equilibrium & collide
    f_[0] += omega * (r * (1.0 / 6.0) * (2.0 - 3.0 * vv) - f_[0]);
    f_[1] += omega * (r * (1.0 / 12.0) * (2.0 + 6.0 * v[0] + 9.0 * vxx - 3.0 * vv) - f_[1]);
    f_[2] += omega * (r * (1.0 / 12.0) * (2.0 - 6.0 * v[0] + 9.0 * vxx - 3.0 * vv) - f_[2]);
    f_[3] += omega * (r * (1.0 / 12.0) * (2.0 + 6.0 * v[1] + 9.0 * vyy - 3.0 * vv) - f_[3]);
    f_[4] += omega * (r * (1.0 / 12.0) * (2.0 - 6.0 * v[1] + 9.0 * vyy - 3.0 * vv) - f_[4]);
    
    // assert!(((f_.sum() + 1e-7) / (r + 1e-7) - 1.0).abs() < 1e-3, "D2Q5 {} {} {}", f_.sum(), r, even);

    // write back to same locations
    if even {
        f[idx[0]][0] = f_[0];
        f[idx[1]][1] = f_[2];
        f[idx[2]][2] = f_[1];
        f[idx[3]][3] = f_[4];
        f[idx[4]][4] = f_[3];
    } else {
        for i in 0..5 {
            f[idx[0]][i] = f_[i];
        }
    }

    val[idx[0]] = r;
}

#[rustfmt::skip]
fn update_d2q9_bgk(
    even: bool,
    omega: f32,
    f: &mut impl ArrayD<VectS<f32, 9>>,
    rho: &mut impl ArrayD<f32>,
    vel: &mut impl ArrayD<VectS<f32, 2>>,
    idx: VectS<i32, 9>,
) {
    // 0:  0  0
    // 1: +1  0
    // 2: -1  0
    // 3:  0 +1
    // 4:  0 -1
    // 5: +1 +1
    // 6: -1 -1
    // 7: -1 +1
    // 8: +1 -1

    // collect fs
    let mut f_ = VectS::new(if even {
        [
            f[idx[0]][0],
            f[idx[1]][1],
            f[idx[2]][2],
            f[idx[3]][3],
            f[idx[4]][4],
            f[idx[5]][5],
            f[idx[6]][6],
            f[idx[7]][7],
            f[idx[8]][8],
        ]
    } else {
        [
            f[idx[0]][0],
            f[idx[0]][2],
            f[idx[0]][1],
            f[idx[0]][4],
            f[idx[0]][3],
            f[idx[0]][6],
            f[idx[0]][5],
            f[idx[0]][8],
            f[idx[0]][7],
        ]
    });

    // calc moments
    let r = f_.sum();
    let vx = (f_[1] - f_[2] + f_[5] - f_[6] - f_[7] + f_[8]) / r;
    let vy = (f_[3] - f_[4] + f_[5] - f_[6] + f_[7] - f_[8]) / r;
    
    let vv = vx * vx + vy * vy;
    let vxx = vx * vx;
    let vyy = vy * vy;
    let vxy = vx * vy;

    // calc equilibrium & collide
    f_[0] += omega * (r * (2.0 / 9.0) * (2.0 - 3.0 * vv) - f_[0]);
    f_[1] += omega * (r * (1.0 / 18.0) * (2.0 + 6.0 * vx + 9.0 * vxx - 3.0 * vv) - f_[1]);
    f_[2] += omega * (r * (1.0 / 18.0) * (2.0 - 6.0 * vx + 9.0 * vxx - 3.0 * vv) - f_[2]);
    f_[3] += omega * (r * (1.0 / 18.0) * (2.0 + 6.0 * vy + 9.0 * vyy - 3.0 * vv) - f_[3]);
    f_[4] += omega * (r * (1.0 / 18.0) * (2.0 - 6.0 * vy + 9.0 * vyy - 3.0 * vv) - f_[4]);
    f_[5] += omega * (r * (1.0 / 36.0) * (1.0 + 3.0 * (vx + vy) + 9.0 * vxy + 3.0 * vv) - f_[5]);
    f_[6] += omega * (r * (1.0 / 36.0) * (1.0 - 3.0 * (vx + vy) + 9.0 * vxy + 3.0 * vv) - f_[6]);
    f_[7] += omega * (r * (1.0 / 36.0) * (1.0 + 3.0 * (vy - vx) - 9.0 * vxy + 3.0 * vv) - f_[7]);
    f_[8] += omega * (r * (1.0 / 36.0) * (1.0 - 3.0 * (vy - vx) - 9.0 * vxy + 3.0 * vv) - f_[8]);
    
    // assert!(((f_.sum() + 1e-7) / (r + 1e-7) - 1.0).abs() < 1e-3, "D2Q9 {} {} {}", f_.sum(), r, even);

    // write back to same locations
    if even {
        f[idx[0]][0] = f_[0];
        f[idx[1]][1] = f_[2];
        f[idx[2]][2] = f_[1];
        f[idx[3]][3] = f_[4];
        f[idx[4]][4] = f_[3];
        f[idx[5]][5] = f_[6];
        f[idx[6]][6] = f_[5];
        f[idx[7]][7] = f_[8];
        f[idx[8]][8] = f_[7];
    } else {
        for i in 0..9 {
            f[idx[0]][i] = f_[i];
        }
    }

    // TODO: we don't actually need to update this until the end of loop_for
    rho[idx[0]] = r;
    vel[idx[0]] = [vx, vy].into();
}

#[repr(C)]
pub enum CellType {
    FLUID = 0,
    WALL = 1,
    FIXED = 2,
}

fn loop_for_advdif_2d(
    iters: usize,
    f: &mut VectDViewVector<f32, 9>,
    rho: &mut VectDView<f32>,
    vel: &mut VectDViewVector<f32, 2>, // in lattice-units
    g: &mut VectDViewVector<f32, 5>,
    conc: &mut VectDView<f32>,
    cells: &VectDView<i32>,
    upstream_idx: &VectDViewVector<i32, 9>,
    counts: VectS<i32, 2>,
    omega_ns: f32, // in lattice-units
    omega_ad: f32, // in lattice-units
) {
    let ncells = counts.prod();

    // Some sanity checks:
    assert_eq!(ncells, f.data_count(), "f");
    assert_eq!(ncells, rho.data_count(), "rho");
    assert_eq!(ncells, vel.data_count(), "vel");
    assert_eq!(ncells, g.data_count(), "g");
    assert_eq!(ncells, conc.data_count(), "conc");

    let iters = if iters % 2 == 0 { iters } else { iters + 1 };

    for iter in 0..iters {
        let even = iter % 2 == 0;

        // let _ = (0..ncells).map(|i| {});

        for idx in 0..ncells {
            if cells[idx] == (CellType::WALL as i32) {
                continue; // implicit bounce-back in AA-update pattern
            }

            let uidx: VectS<i32, 9> = upstream_idx[idx];
            if cells[idx] == (CellType::FIXED as i32) {
                update_d2q9_fixed(even, f, rho, vel, uidx);
                update_d2q5_fixed(even, g, conc, vel, uidx);
            } else {
                update_d2q9_bgk(even, omega_ns, f, rho, vel, uidx);
                update_d2q5_bgk(even, omega_ad, g, conc, vel, uidx);
            }
        }
    }
}

/// See: opencl3 example [`basic.rs`](https://github.com/kenba/opencl3/blob/main/examples/basic.rs)
fn loop_for_advdif_2d_opencl(
    iters: usize,
    ctx: &OpenCLCtx,
    fluid: &mut Fluid,
    scalar: &mut Scalar,
    cells: &Cells,
    counts: VectS<i32, 2>, // TODO: can't I just store this on the context?
    omega_ns: f32,         // in lattice-units
    omega_ad: f32,         // in lattice-units
) {
    let queue = &ctx.queue;
    let d2q9 = &ctx.d2q9_kernel;
    let d2q5 = &ctx.d2q5_kernel;

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
                .set_global_work_size(counts.prod() as usize)
                .enqueue_nd_range(&queue)
                .expect("ExecuteKernel::new failed.")
        };
    }

    for iter in 0..iters {
        let even = (1 - iter % 2) as i32;

        // let _ = unsafe {
        //     exec_kernel!(
        //         d2q9,
        //         &even,
        //         &omega_ns,
        //         &mut fluid.f,
        //         &mut fluid.rho,
        //         &mut fluid.vel,
        //         &cells.cell_type,
        //         &cells.offset_idx
        //     )
        // };

        // let _ = unsafe {
        //     exec_kernel!(
        //         d2q5,
        //         &even,
        //         &omega_ad,
        //         &mut scalar.f,
        //         &mut scalar.val,
        //      &fluid.vel
        //         &cells.cell_type,
        //         &cells.offset_idx
        //     )
        // };

        let d2q9_event = unsafe {
            ExecuteKernel::new(&d2q9)
                .set_arg(&even)
                .set_arg(&omega_ns)
                .set_arg(&mut fluid.f)
                .set_arg(&mut fluid.rho)
                .set_arg(&mut fluid.vel)
                .set_arg(&cells.cell_type)
                .set_arg(&cells.offset_idx)
                .set_global_work_size(counts.prod() as usize)
                .enqueue_nd_range(&queue)
                .expect("ExecuteKernel::new failed.")
        };

        events.push(d2q9_event.get());
        queue.finish().expect("queue.finish failed");

        let d2q5_event = unsafe {
            ExecuteKernel::new(&d2q5)
                .set_arg(&even)
                .set_arg(&omega_ad)
                .set_arg(&mut scalar.f)
                .set_arg(&mut scalar.val)
                .set_arg(&fluid.vel)
                .set_arg(&cells.cell_type)
                .set_arg(&cells.offset_idx)
                .set_global_work_size(counts.prod() as usize)
                .enqueue_nd_range(&queue)
                .expect("ExecuteKernel::new failed.")
        };

        events.push(d2q5_event.get());
        queue.finish().expect("queue.finish failed");
    }
}

/// Return an array of size (cell_count, Q) where for each cell we have calculated the index of the
/// cell upstream of each velocity in the model.
pub fn upstream_idx<const D: usize, const Q: usize, const ROW_MAJOR: bool>(
    counts: Counts<D>,
    model: Model<D, Q>,
) -> VectD<VectS<i32, Q>> {
    let strides = counts_to_strides::<D, ROW_MAJOR>(counts);
    let mut idx: VectD<VectS<i32, Q>> = VectD::zeros(counts.0.prod() as usize);
    let mut i = 0;
    for sub in Raster::<D, ROW_MAJOR>::new(counts) {
        for q in 0..Q {
            let sub_ = sub.0 - model.q[q].cast();
            let sub_ = vmod(&sub_, &counts.0);
            let j = sub_to_idx(Sub(sub_), strides);
            idx[i][q] = j;
        }

        i += 1;
    }

    return idx;
}

const OPENCL_SRC: &str = include_str!("lib.cl");

/// Long-lived OpenCL context.
/// Exists so we do not need to recreate everthing for each batch of iterations.
struct OpenCLCtx {
    _device: Device,
    context: Context,
    queue: CommandQueue,
    // TODO: Need to make this more generic to handle different sim requirements.
    d2q9_kernel: Kernel,
    d2q5_kernel: Kernel,
}

#[pyclass(name = "OpenCLCtx")]

struct OpenCLCtxPy {
    ctx: Arc<Mutex<OpenCLCtx>>,
}

#[pymethods]
impl OpenCLCtxPy {
    #[new]
    fn new() -> Self {
        let device_id = *get_all_devices(CL_DEVICE_TYPE_DEFAULT) //CL_DEVICE_TYPE_GPU
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
            ctx: Arc::new(Mutex::new(OpenCLCtx {
                _device: device,
                context: context,
                queue: queue,
                d2q9_kernel: d2q9,
                d2q5_kernel: d2q5,
            })),
        }
    }
}

/// Long-lived container for OpenCL device buffers.
/// Exists so we do not need to recreate the buffers for each batch of iterations.
#[pyclass]
struct Fluid {
    f: Buffer<f32>,
    rho: Buffer<f32>,
    vel: Buffer<f32>,
}

/// Construct an OpenCL [`Buffer`] from the passed [`VectDView`].
fn make_buffer<T: Data>(ctx: &Context, arr_host: &VectDView<T>) -> Buffer<T::Elem> {
    unsafe {
        Buffer::create(
            ctx,
            CL_MEM_READ_WRITE,
            arr_host.elem_count() as usize,
            ptr::null_mut(),
        )
        .expect("Buffer::create() failed")
    }
}

/// Write our host array to the OpenCL [`Buffer`].
fn enqueue_write<T: Data>(
    queue: &CommandQueue,
    arr_host: &VectDView<T>,
    arr_dev: &mut Buffer<T::Elem>,
) -> Event {
    unsafe {
        queue
            .enqueue_write_buffer(
                arr_dev,
                CL_BLOCKING,
                0,
                arr_host.as_slice(),
                &[], // ignore events ... why?
            )
            .expect("enqueue_write_buffer failed")
    }
}

// Read back from OpenCL device buffers into our host arrays.
fn enqueue_read<T: Data>(
    queue: &CommandQueue,
    arr_host: &mut VectDView<T>,
    arr_dev: &Buffer<T::Elem>,
) -> Event {
    unsafe {
        queue
            .enqueue_read_buffer(
                arr_dev,
                CL_BLOCKING,
                0,
                arr_host.as_mut_slice(),
                &[], // ignore events ... why?
            )
            .expect("enqueue_read_buffer failed")
    }
}

#[pymethods]
impl Fluid {
    #[new]
    pub fn new(
        cl: Bound<'_, OpenCLCtxPy>,
        f: PyReadwriteArray2<f32>,
        rho: PyReadwriteArray1<f32>,
        vel: PyReadwriteArray2<f32>,
    ) -> Self {
        // Construct views into the numpy arrays.
        let f = VectDViewVector::<f32, 9>::from(f);
        let rho = VectDViewScalar::<f32>::from(rho);
        let vel = VectDViewVector::<f32, 2>::from(vel);

        // Make the OpenCL buffers.
        let cl = cl.borrow();
        let cl = cl.ctx.lock().unwrap();
        let ctx = &cl.context;
        let queue = &cl.queue;

        let mut out = Self {
            f: make_buffer(ctx, &f),
            rho: make_buffer(ctx, &rho),
            vel: make_buffer(ctx, &vel),
        };

        // Copy data into the buffers
        let _write_f = enqueue_write(queue, &f, &mut out.f);
        let _write_r = enqueue_write(queue, &rho, &mut out.rho);
        let _write_v = enqueue_write(queue, &vel, &mut out.vel);

        queue.finish().expect("queue.finish() failed [Fluid]");

        out
    }

    /// Read the data back from the OpenCL buffers into our numpy arrays.
    pub fn read(
        &self,
        cl: Bound<'_, OpenCLCtxPy>,
        f: PyReadwriteArray2<f32>,
        rho: PyReadwriteArray1<f32>,
        vel: PyReadwriteArray2<f32>,
    ) {
        // Construct views into the numpy arrays.
        let mut f = VectDViewVector::<f32, 9>::from(f);
        let mut rho = VectDViewScalar::<f32>::from(rho);
        let mut vel = VectDViewVector::<f32, 2>::from(vel);

        // Copy data into the arrays
        let cl = cl.borrow();
        let cl = cl.ctx.lock().unwrap();
        let queue = &cl.queue;

        let _read_f = enqueue_read(queue, &mut f, &self.f);
        let _read_r = enqueue_read(queue, &mut rho, &self.rho);
        let _read_v = enqueue_read(queue, &mut vel, &self.vel);
    }
}

/// Long-lived container for OpenCL device buffers.
///
/// Exists so we do not need to recreate the buffers for each batch of iterations.
#[pyclass]
struct Scalar {
    f: Buffer<f32>,
    val: Buffer<f32>,
}

#[pymethods]
impl Scalar {
    #[new]
    pub fn new(
        cl: Bound<'_, OpenCLCtxPy>,
        f: PyReadwriteArray2<f32>,
        val: PyReadwriteArray1<f32>,
    ) -> Self {
        // Construct views into the numpy arrays.
        let f = VectDViewVector::<f32, 5>::from(f);
        let val = VectDViewScalar::<f32>::from(val);

        // Make the OpenCL buffers.
        let cl = cl.borrow();
        let cl = cl.ctx.lock().unwrap();
        let ctx = &cl.context;
        let queue = &cl.queue;

        let mut out = Self {
            f: make_buffer(ctx, &f),
            val: make_buffer(ctx, &val),
        };

        // Copy data into the buffers
        let _write_f = enqueue_write(queue, &f, &mut out.f);
        let _write_r = enqueue_write(queue, &val, &mut out.val);

        queue.finish().expect("queue.finish() failed [Scalar]");

        out
    }

    /// Read the data back from the OpenCL buffers into our numpy arrays.
    pub fn read(
        &self,
        cl: Bound<'_, OpenCLCtxPy>,
        f: PyReadwriteArray2<f32>,
        val: PyReadwriteArray1<f32>,
    ) {
        // Construct views into the numpy arrays.
        let mut f = VectDViewVector::<f32, 5>::from(f);
        let mut val = VectDViewScalar::<f32>::from(val);

        // Copy data into the arrays
        let cl = cl.borrow();
        let cl = cl.ctx.lock().unwrap();
        let queue = &cl.queue;

        let _read_f = enqueue_read(queue, &mut f, &self.f);
        let _read_c = enqueue_read(queue, &mut val, &self.val);
    }
}

#[pyclass]
struct Cells {
    /// The type of the cell as given by [`CellType`].
    cell_type: Buffer<i32>,

    /// The _upstream_ cell indices in each direction.
    offset_idx: Buffer<i32>,
}

#[pymethods]
impl Cells {
    #[new]
    pub fn new(
        cl: Bound<'_, OpenCLCtxPy>,
        cell_types: PyReadwriteArray1<i32>,
        offset_idx: PyReadwriteArray2<i32>,
    ) -> Self {
        // Construct views into the numpy arrays.
        let cell_type = VectDViewScalar::<i32>::from(cell_types);
        let offset_idx = VectDViewVector::<i32, 9>::from(offset_idx);

        // Make the OpenCL buffers.
        let cl = cl.borrow();
        let cl = cl.ctx.lock().unwrap();
        let ctx = &cl.context;
        let queue = &cl.queue;

        let mut out = Self {
            cell_type: make_buffer(ctx, &cell_type),
            offset_idx: make_buffer(ctx, &offset_idx),
        };

        // Copy data into the buffers
        let _c_write = enqueue_write(queue, &cell_type, &mut out.cell_type);
        let _i_write = enqueue_write(queue, &offset_idx, &mut out.offset_idx);

        queue.finish().expect("queue.finish() failed [Cells]");

        out
    }

    /// Read the data back from the OpenCL buffers into our numpy arrays.
    pub fn read(
        &self,
        cl: Bound<'_, OpenCLCtxPy>,
        cell_types: PyReadwriteArray1<i32>,
        offset_idx: PyReadwriteArray2<i32>,
    ) {
        // Construct views into the numpy arrays.
        let mut cell_type = VectDViewScalar::<i32>::from(cell_types);
        let mut offset_idx = VectDViewVector::<i32, 9>::from(offset_idx);

        // Copy data into the arrays
        let cl = cl.borrow();
        let cl = cl.ctx.lock().unwrap();
        let queue = &cl.queue;

        let _read_c = enqueue_read(queue, &mut cell_type, &self.cell_type);
        let _read_i = enqueue_read(queue, &mut offset_idx, &self.offset_idx);
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn boltzmann_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(name = "calc_curl_2d")]
    fn calc_curl_2d_py<'py>(
        _py: Python<'py>,
        vel: PyReadwriteArray2<f32>,
        cells: PyReadwriteArray1<i32>,
        indices: PyReadwriteArray2<i32>,
        curl: PyReadwriteArray1<f32>,
    ) {
        calc_curl_2d(
            &VectDView::from(vel),
            &VectDView::from(cells),
            &VectDView::from(indices),
            &mut VectDView::from(curl),
        );
    }

    #[pyfn(m)]
    #[pyo3(name = "loop_for_advdif_2d")]
    fn loop_for_advdif_2d_py<'py>(
        _py: Python<'py>,
        iters: usize,
        f: PyReadwriteArray2<f32>,
        rho: PyReadwriteArray1<f32>,
        vel: PyReadwriteArray2<f32>, // in lattice-units
        g: PyReadwriteArray2<f32>,
        conc: PyReadwriteArray1<f32>,
        cells: PyReadwriteArray1<i32>,
        upstream_idx: PyReadwriteArray2<i32>,
        counts: PyReadwriteArray1<i32>,
        omega_ns: f32, // in lattice-units
        omega_ad: f32, // in lattice-units
    ) -> PyResult<()> {
        loop_for_advdif_2d(
            iters,
            &mut VectDView::from(f),
            &mut VectDView::from(rho),
            &mut VectDView::from(vel),
            &mut VectDView::from(g),
            &mut VectDView::from(conc),
            &VectDView::from(cells),
            &VectDView::from(upstream_idx),
            VectS::<i32, 2>::from(counts),
            omega_ns,
            omega_ad,
        );

        return Ok(());
    }

    #[pyfn(m)]
    #[pyo3(name = "loop_for_advdif_2d_opencl")]
    fn loop_for_advdif_2d_opencl_py<'py>(
        _py: Python<'py>,
        iters: usize,
        ctx: Bound<'py, OpenCLCtxPy>,
        fluid: Bound<'py, Fluid>,
        scalar: Bound<'py, Scalar>,
        cells: Bound<'py, Cells>,
        counts: PyReadwriteArray1<i32>,
        omega_ns: f32, // in lattice-units
        omega_ad: f32, // in lattice-units
    ) -> PyResult<()> {
        // Aquire context lock
        let ctx = ctx.borrow();
        let ctx = ctx
            .ctx
            .lock()
            .expect("Unable to aquire OpenCL context lock.");

        loop_for_advdif_2d_opencl(
            iters,
            &ctx,
            &mut fluid.borrow_mut(),
            &mut scalar.borrow_mut(),
            &mut cells.borrow_mut(),
            VectS::<i32, 2>::from(counts),
            omega_ns,
            omega_ad,
        );

        return Ok(());
    }

    m.add_class::<OpenCLCtxPy>()?;
    m.add_class::<Fluid>()?;
    m.add_class::<Scalar>()?;
    m.add_class::<Cells>()?;

    Ok(())
}
