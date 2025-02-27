use ndarray::{arr1, arr2, Array1, Array2, ArrayView1, Axis};
use serde::{Deserialize, Serialize};

use crate::{
    raster::{counts_to_strides, raster_row_major, sub_to_idx, Ix, StrideOrder},
    utils::vmod,
};

/// Contains the weights & velocities for calculating offsets and equilibrium distribution functions.
///
/// TODO: This is a bit duplicated with `kernel_gen.py`, we could also generate the OpenCL
///       calc_equilibrium and do it device side before iterating. That would be nicer.
///       It's also completely gross that I just hardcoded the values.
#[derive(Clone, Serialize, Deserialize)]
pub struct VelocitySet {
    pub ws: Array1<f32>,
    pub qs: Array2<i32>,
}

macro_rules! ax0 {
    ($arr:expr, $idx:expr) => {
        ($arr).index_axis(Axis(0), $idx)
    };
}

// expose ax0 outside this module but within the crate
// pub(crate) use ax0;

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
    pub fn feq(&self, rho: f32, vel: ArrayView1<f32>) -> Array1<f32> {
        let mut out: Array1<f32> = Array1::zeros([self.Q()]);
        for q in 0..self.Q() {
            let q_ = ax0!(self.qs, q).mapv(|x| x as f32);
            let vq = vel.dot(&q_);
            let vv = vel.dot(&vel);
            out[q] = rho * self.ws[q] * (1.0 + 3.0 * vq + 4.5 * vq * vq - (3.0 / 2.0) * vv)
        }
        out
    }

    // PONDER: can't this just be an enum?
    pub fn make(d: usize, q: usize) -> VelocitySet {
        match (d, q) {
            (1, 3) => Self::D1Q3(),
            (2, 9) => Self::D2Q9(),
            (2, 5) => Self::D2Q5(),
            (3, 27) => Self::D3Q27(),
            _ => panic!("Unknown model: D{}Q{}", d, q),
        }
    }

    pub fn D1Q3() -> VelocitySet {
        VelocitySet {
            ws: arr1(&[4. / 6., 1. / 6., 1. / 6.]),
            qs: arr2(&[[0], [1], [-1]]),
        }
    }

    pub fn D2Q9() -> VelocitySet {
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

    pub fn D2Q5() -> VelocitySet {
        VelocitySet {
            ws: arr1(&[1. / 3., 1. / 6., 1. / 6., 1. / 6., 1. / 6.]),
            qs: arr2(&[[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]]),
        }
    }

    pub fn D3Q27() -> VelocitySet {
        VelocitySet {
            ws: arr1(&[
                8. / 27.,
                2. / 27.,
                2. / 27.,
                2. / 27.,
                1. / 54.,
                1. / 54.,
                2. / 27.,
                1. / 54.,
                1. / 54.,
                2. / 27.,
                1. / 54.,
                1. / 54.,
                1. / 54.,
                1. / 216.,
                1. / 216.,
                1. / 54.,
                1. / 216.,
                1. / 216.,
                2. / 27.,
                1. / 54.,
                1. / 54.,
                1. / 54.,
                1. / 216.,
                1. / 216.,
                1. / 54.,
                1. / 216.,
                1. / 216.,
            ]),
            qs: arr2(&[
                [0, 0, 0],
                [0, 0, 1],
                [0, 0, -1],
                [0, 1, 0],
                [0, 1, 1],
                [0, 1, -1],
                [0, -1, 0],
                [0, -1, 1],
                [0, -1, -1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 0, -1],
                [1, 1, 0],
                [1, 1, 1],
                [1, 1, -1],
                [1, -1, 0],
                [1, -1, 1],
                [1, -1, -1],
                [-1, 0, 0],
                [-1, 0, 1],
                [-1, 0, -1],
                [-1, 1, 0],
                [-1, 1, 1],
                [-1, 1, -1],
                [-1, -1, 0],
                [-1, -1, 1],
                [-1, -1, -1],
            ]),
        }
    }

    /// Return an array of size (cell_count, q) where for each cell we have calculated the index of the
    /// cell upstream of each velocity in the model.
    pub fn upstream_idx(&self, counts: &Array1<Ix>) -> Array2<i32> {
        let strides = counts_to_strides(counts, StrideOrder::RowMajor);
        let mut idx = Array2::zeros([counts.product() as usize, self.Q()]);
        let mut i = 0;
        for sub in raster_row_major(counts.clone()) {
            for q in 0..self.Q() {
                let sub_ = sub.clone() + ax0!(self.qs, q).mapv(|q| q as Ix);
                let sub_ = vmod(sub_, &counts);
                let j = sub_to_idx(&sub_, &strides);
                idx[(i, q)] = j as Ix;
            }

            i += 1;
        }

        return idx;
    }
}
