use std::ptr::slice_from_raw_parts_mut;
use std::{mem, str, usize};

use lbm::{tensor_prod_q, tensor_prod_w, D1Q3_Q, D1Q3_W, LBM};
use num_traits::Zero;
use numpy::ndarray::{arr1, s, Array1, Array2, ArrayViewMut1, ArrayViewMut2};
// use numpy::{convert::IntoPyArray, ndarray::Dim, npyffi::npy_intp, PyArray, ToPyArray};
use numpy::{PyArray1, PyReadonlyArray1, PyReadwriteArray1, PyReadwriteArray2};
use pyo3::prelude::*;
use raster::{sub_to_idx, Raster};
use utils::vmod;
use vect_d::{ArrayD, VectD, VectDView};
use vect_s::VectS;

mod lbm;
mod raster;
mod utils;
mod vect_d;
mod vect_s;

/// PONDER: Making this generic over D & Q gives us some nice compile-time checks, but it makes
///         exposing to Python much more clunky. Is it worth doing away with static D & Q and making
///         them runtime? I wonder if there are also any perf benefits from knowing them at
///         compile-time?
struct Model<const D: usize, const Q: usize> {
    w: [f32; Q],
    q: [VectS<i32, D>; Q],
}

impl<const D: usize, const Q: usize> Model<D, Q> {
    const D: usize = D;
    const Q: usize = Q;
}

static D2Q9: Model<2, 9> = Model::<2, 9> {
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

struct LatticeMeta<const D: usize> {
    counts: VectS<i32, D>,
}

impl<const D: usize> From<[i32; D]> for LatticeMeta<D> {
    fn from(value: [i32; D]) -> Self {
        Self {
            counts: value.into(),
        }
    }
}

struct Field<const D: usize, const Q: usize> {
    name: String,
    model: &'static Model<D, Q>,
    f: VectD<VectS<f32, Q>>,
}

impl<const D: usize, const Q: usize> Field<D, Q> {
    pub fn new(
        name: impl Into<String>,
        model: &'static Model<D, Q>,
        lattice: &LatticeMeta<D>,
    ) -> Field<D, Q> {
        Self {
            name: name.into(),
            model: model,
            f: VectD::zeros(lattice.counts.prod() as usize),
        }
    }
}

#[pyclass]
struct LatticeMeta2D {
    meta: LatticeMeta<2>,
}

#[pymethods]
impl LatticeMeta2D {
    #[getter]
    pub fn counts(&self) -> PyResult<[i32; 2]> {
        Ok(self.meta.counts.into())
    }
}

#[pyclass]
struct FieldD2Q9 {
    field: Field<2, 9>,
}

#[pymethods]
impl FieldD2Q9 {
    #[new]
    pub fn new(name: String, lattice: &LatticeMeta2D) -> FieldD2Q9 {
        Self {
            field: Field::new(name, &D2Q9, &lattice.meta),
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
    idx: VectS<i32, 9>,
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
    f_[0] += omega * (r * (2.0 / 9.0) * (2.0 - 3.0 * vv) - f_[0]);
    f_[1] += omega * (r * (1.0 / 18.0) * (2.0 + 6.0 * v[0] + 9.0 * vxx - 3.0 * vv) - f_[1]);
    f_[2] += omega * (r * (1.0 / 18.0) * (2.0 - 6.0 * v[0] + 9.0 * vxx - 3.0 * vv) - f_[2]);
    f_[3] += omega * (r * (1.0 / 18.0) * (2.0 + 6.0 * v[1] + 9.0 * vyy - 3.0 * vv) - f_[3]);
    f_[4] += omega * (r * (1.0 / 18.0) * (2.0 - 6.0 * v[1] + 9.0 * vyy - 3.0 * vv) - f_[4]);
    
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
    let vx = (f_[1] + f_[5] + f_[8] - f_[3] - f_[6] - f_[7]) / r;
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

    rho[idx[0]] = r;
    vel[idx[0]] = [vx, vy].into();
}

fn loop_for_advdif_2d(
    iters: usize,
    f: &mut impl ArrayD<VectS<f32, 9>>,
    rho: &mut impl ArrayD<f32>,
    vel: &mut impl ArrayD<VectS<f32, 2>>, // in lattice-units
    g: &mut impl ArrayD<VectS<f32, 5>>,
    conc: &mut impl ArrayD<f32>,
    is_wall: &impl ArrayD<bool>,
    is_fixed: &impl ArrayD<bool>,
    upstream_idx: &impl ArrayD<VectS<i32, 9>>,
    counts: VectS<i32, 2>,
    omega_ns: f32, // in lattice-units
    omega_ad: f32, // in lattice-units
) {
    let ncells = counts.prod();

    // Some sanity checks:
    assert_eq!(ncells, f.len());
    assert_eq!(ncells, rho.len());
    assert_eq!(ncells, vel.len());
    assert_eq!(ncells, g.len());
    assert_eq!(ncells, conc.len());

    let iters = if iters % 2 == 0 { iters } else { iters + 1 };

    for iter in 0..iters {
        let even = iter % 2 == 0;

        for idx in 0..ncells {
            let no_update = is_wall[idx] || is_fixed[idx];

            // TODO: streaming from fixed but not wall

            if no_update {
                continue;
            }

            let uidx = upstream_idx[idx];
            update_d2q9_bgk(even, omega_ns, f, rho, vel, uidx);
            update_d2q5_bgk(even, omega_ad, g, conc, vel, uidx);
        }
    }
}

/// Return an array of size (cell_count, Q) where for each cell we have calculated the index of the
/// cell upstream of each velocity in the model.
fn upstream_idx<const D: usize, const Q: usize>(
    counts: VectS<i32, D>,
    model: Model<D, Q>,
) -> VectD<VectS<i32, Q>> {
    let mut idx: VectD<VectS<i32, Q>> = VectD::zeros(counts.prod() as usize);
    let mut i = 0;
    for sub in Raster::new(counts) {
        for q in 0..Q {
            let sub_ = sub - model.q[q].cast();
            let sub_ = vmod(sub_, counts);
            let j = sub_to_idx(sub_, counts);
            idx[i][q] = j;
        }

        i += 1;
    }

    return idx;
}

fn test_1(x: &mut impl ArrayD<f32>) {
    x[1] = 1.0;
}

fn test_2(x: &mut impl ArrayD<VectS<f32, 2>>) {
    x[1][0] = 1.0;
    x[1][1] = 42.0;
}

/// A Python module implemented in Rust.
#[pymodule]
fn boltzmann_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(name = "test_1")]
    fn test_1_py<'py>(_py: Python<'py>, x: PyReadwriteArray1<f32>) -> PyResult<()> {
        let mut x_: VectDView<f32> = x.into();
        test_1(&mut x_);

        return Ok(());
    }

    #[pyfn(m)]
    #[pyo3(name = "test_2")]
    fn test_2_py<'py>(_py: Python<'py>, x: PyReadwriteArray2<f32>) -> PyResult<()> {
        let mut x_: VectDView<VectS<f32, 2>> = x.into();
        test_2(&mut x_);

        return Ok(());
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
        is_wall: PyReadwriteArray1<bool>,
        is_fixed: PyReadwriteArray1<bool>,
        upstream_idx: PyReadwriteArray2<i32>,
        counts: PyReadwriteArray1<i32>,
        omega_ns: f32, // in lattice-units
        omega_ad: f32, // in lattice-units
    ) -> PyResult<()> {
        loop_for_advdif_2d(
            iters,
            &mut VectDView::<VectS<f32, 9>>::from(f),
            &mut VectDView::<f32>::from(rho),
            &mut VectDView::<VectS<f32, 2>>::from(vel),
            &mut VectDView::<VectS<f32, 5>>::from(g),
            &mut VectDView::<f32>::from(conc),
            &VectDView::<bool>::from(is_wall),
            &VectDView::<bool>::from(is_fixed),
            &VectDView::<VectS<i32, 9>>::from(upstream_idx),
            VectS::<i32, 2>::from(counts),
            omega_ns,
            omega_ad,
        );

        return Ok(());
    }

    m.add_class::<LatticeMeta2D>()?;
    m.add_class::<FieldD2Q9>()?;

    Ok(())
}
