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
enum CellType {
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

const UPDATE_D2Q9_BGK_SRC: &str = include_str!("lib.cl");

/// See: opencl3 example [`basic.rs`](https://github.com/kenba/opencl3/blob/main/examples/basic.rs)
pub fn loop_for_advdif_2d_opencl(
    iters: usize,
    f: &mut VectDViewVector<f32, 9>,
    rho: &mut VectDViewScalar<f32>,
    vel: &mut VectDViewVector<f32, 2>, // in lattice-units
    g: &mut VectDViewVector<f32, 5>,
    conc: &mut VectDViewScalar<f32>,
    cell: &VectDViewScalar<i32>,
    upstream_idx: &VectDViewVector<i32, 9>,
    counts: VectS<i32, 2>,
    omega_ns: f32, // in lattice-units
    omega_ad: f32, // in lattice-units
) {
    let device_id = *get_all_devices(CL_DEVICE_TYPE_DEFAULT) //CL_DEVICE_TYPE_GPU
        .expect("error getting platform")
        .first()
        .expect("no device found in platform");
    let device = Device::new(device_id);
    let context = Context::from_device(&device).expect("Context::from_device failed");
    let queue = CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE)
        .expect("CommandQueue::create_default failed");

    let program = Program::create_and_build_from_source(&context, UPDATE_D2Q9_BGK_SRC, "")
        .expect("Program::create_and_build_from_source failed");
    let kernel = Kernel::create(&program, "update_d2q9_bgk").expect("Kernel::create failed");

    // NOTE: Lack of generic closures also makes this a tad verbose because we must pass `&context`.

    fn make_buffer<T: Data>(context: &Context, arr: &VectDView<T>) -> Buffer<T::Elem> {
        unsafe {
            Buffer::create(
                context,
                CL_MEM_READ_WRITE,
                arr.elem_count() as usize,
                ptr::null_mut(),
            )
            .expect("Buffer::create() failed.")
        }
    }

    // Create OpenCL device buffers & write our data to them.
    let mut f_dev = make_buffer(&context, f);
    let mut rho_dev = make_buffer(&context, rho);
    let mut vel_dev = make_buffer(&context, vel);
    let mut idx_dev = make_buffer(&context, upstream_idx);
    let mut cell_dev = make_buffer(&context, cell);

    macro_rules! enqueue_write {
        ($arr_from:ident, $arr_to:ident) => {
            unsafe {
                queue
                    .enqueue_write_buffer(
                        &mut $arr_to,
                        CL_BLOCKING,
                        0,
                        (&($arr_from)).as_slice(),
                        &[], // ignore events ... why?
                    )
                    .expect(&*format!(
                        "enqueue_write_buffer failed {}",
                        stringify!($arr_from)
                    ))
            }
        };
    }

    let _f_write = enqueue_write!(f, f_dev);
    let _rho_write = enqueue_write!(rho, rho_dev);
    let _vel_write = enqueue_write!(vel, vel_dev);
    let _idx_write = enqueue_write!(upstream_idx, idx_dev);
    let _cell_write = enqueue_write!(cell, cell_dev);

    let mut events: Vec<cl_event> = Vec::default();

    for iter in 0..iters {
        let even = (1 - iter % 2) as i32;

        let kernel_event = unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(&even)
                .set_arg(&omega_ns)
                .set_arg(&mut f_dev)
                .set_arg(&mut rho_dev)
                .set_arg(&mut vel_dev)
                .set_arg(&cell_dev)
                .set_arg(&idx_dev)
                .set_global_work_size(counts.prod() as usize)
                // .set_wait_event(&f_write)
                // .set_wait_event(&rho_write)
                // .set_wait_event(&vel_write)
                // .set_wait_event(&idx_write)
                // .set_wait_event(&cell_write)
                .enqueue_nd_range(&queue)
                .expect("ExecuteKernel::new failed.")
        };

        events.push(kernel_event.get());

        // Wait for kernel to execute.
        queue.finish().expect("queue.finish failed");

        // Read back from OpenCL device buffers into our host arrays.
        fn enqueue_read<T: Data>(
            queue: &CommandQueue,
            events: &[cl_event],
            buf: &Buffer<T::Elem>,
            arr: &mut VectDView<T>,
        ) -> Event {
            unsafe {
                queue
                    .enqueue_read_buffer(buf, CL_BLOCKING, 0, arr.as_mut_slice(), events)
                    .expect("enqueue_read_buffer failed f")
            }
        }

        let _read_f = enqueue_read(&queue, &[], &f_dev, f);
        let _read_rho = enqueue_read(&queue, &[], &rho_dev, rho);
        let _read_vel = enqueue_read(&queue, &[], &vel_dev, vel);

        // Wait for the read events to complete.
        // read_f.wait().expect("Wait failed.");
        // read_rho.wait().expect("Wait failed.");
        // read_vel.wait().expect("Wait failed.");

        // Do Advection-Diffusion the old-school way
        for idx in 0..counts.prod() {
            let uidx: VectS<i32, 9> = upstream_idx[idx];
            if cell[idx] == 2 {
                update_d2q5_fixed(even != 0, g, conc, vel, uidx);
            } else {
                update_d2q5_bgk(even != 0, omega_ad, g, conc, vel, uidx);
            }
        }
    }
}

/// Return an array of size (cell_count, Q) where for each cell we have calculated the index of the
/// cell upstream of each velocity in the model.
fn upstream_idx<const D: usize, const Q: usize, const ROW_MAJOR: bool>(
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
        f: PyReadwriteArray2<f32>,
        rho: PyReadwriteArray1<f32>,
        vel: PyReadwriteArray2<f32>, // in lattice-units
        g: PyReadwriteArray2<f32>,
        conc: PyReadwriteArray1<f32>,
        cell: PyReadwriteArray1<i32>,
        upstream_idx: PyReadwriteArray2<i32>,
        counts: PyReadwriteArray1<i32>,
        omega_ns: f32, // in lattice-units
        omega_ad: f32, // in lattice-units
    ) -> PyResult<()> {
        loop_for_advdif_2d_opencl(
            iters,
            &mut VectDView::<VectS<f32, 9>>::from(f),
            &mut VectDView::<f32>::from(rho),
            &mut VectDView::<VectS<f32, 2>>::from(vel),
            &mut VectDView::<VectS<f32, 5>>::from(g),
            &mut VectDView::<f32>::from(conc),
            &VectDView::<i32>::from(cell),
            &VectDView::<VectS<i32, 9>>::from(upstream_idx),
            VectS::<i32, 2>::from(counts),
            omega_ns,
            omega_ad,
        );

        return Ok(());
    }

    Ok(())
}
