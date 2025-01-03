use crate::raster::{raster_row_major, sub_to_idx, Raster};
use crate::utils::vmod;
use crate::vect_d::VectD;
use crate::vect_s::VectS;
use num_traits::Zero;

/// Calculate the approximate equilibrium distribution for the given density,
/// velocity & lattice weight/velocity set.
fn calc_f_eq<const D: usize, const Q: usize>(
    rho: f32,
    vel: VectS<f32, D>,
    ws: VectS<f32, Q>,
    qs: [VectS<f32, D>; Q],
) -> VectS<f32, Q> {
    let vv = (vel * vel).sum();
    let mut out = ws;
    for i in 0..Q {
        let vq = (vel * qs[i]).sum();
        out[i] *= rho * (1.0 + 3.0 * vq - 1.5 * vv + 4.5 * vq * vq);
    }
    out
}

fn update_generic_bgk<const D: usize, const Q: usize>(
    even: bool,
    omega: f32,
    f: &mut VectD<VectS<f32, Q>>,
    rho: &mut VectD<f32>,
    vel: &mut VectD<VectS<f32, D>>,
    idx: VectS<i32, Q>,
    ws: VectS<f32, Q>,
    qs: [VectS<f32, D>; Q],
    js: [usize; Q],
) {
    // collect fs
    let mut f_: VectS<f32, Q> = VectS::default();
    for i in 0..Q {
        f_[i] = if even { f[idx[i]][i] } else { f[idx[0]][js[i]] };
    }

    // calc moments
    let r = f_.sum();
    let mut v = VectS::<f32, D>::zero();
    for i in 0..Q {
        v += f_[i] * qs[i] / r;
    }
    let v = v; // no mut
    let vv = (v * v).sum();

    // calc equilibrium & collide
    for i in 0..Q {
        let vq = (v * qs[i]).sum();
        let feq = r * ws[i] * (1.0 + 3.0 * vq - 1.5 * vv + 4.5 * vq * vq);
        f_[i] += omega * (feq - f_[i]);
    }
    // let feq = calc_f_eq(r, v, ws, qs);
    // f_ += omega * (feq - f_);

    // write back to same locations
    for i in 0..Q {
        if even {
            let j = js[i];
            f[idx[j]][j] = f_[i];
        } else {
            f[idx[0]][i] = f_[i];
        }
    }

    // update the macroscopic observables
    rho[idx[0]] = r;
    vel[idx[0]] = v;
}

fn update_d1q3_bgk<const D: usize, const Q: usize>(
    even: bool,
    omega: f32,
    f: &mut VectD<VectS<f32, Q>>,
    rho: &mut VectD<f32>,
    vel: &mut VectD<VectS<f32, D>>,
    idx: VectS<i32, Q>,
) {
    // Boo; we have to make this function generic but only want D1Q3.
    assert_eq!(D, 1);
    assert_eq!(Q, 3);

    // collect fs
    let (f0, f1, f2) = if even {
        (f[idx[0]][0], f[idx[1]][1], f[idx[2]][2])
    } else {
        (f[idx[0]][0], f[idx[0]][2], f[idx[0]][1])
    };

    // calc moments
    let r = f0 + f1 + f2;
    let v = (f1 - f2) / r;
    let vv = v * v;

    // calc equilibrium
    let f0eq = r * (1. / 3.) * (2. - 3. * vv);
    let f1eq = r * (1. / 12.) * (2. + 6. * v + 6. * vv);
    let f2eq = r * (1. / 12.) * (2. - 6. * v + 6. * vv);

    // write back to same locations
    if even {
        f[idx[0]][0] = f0 + omega * (f0eq - f0);
        f[idx[2]][2] = f1 + omega * (f1eq - f1);
        f[idx[1]][1] = f2 + omega * (f2eq - f2);
    } else {
        f[idx[0]][0] = f0 + omega * (f0eq - f0);
        f[idx[0]][1] = f1 + omega * (f1eq - f1);
        f[idx[0]][2] = f2 + omega * (f2eq - f2);
    }

    rho[idx[0]] = r;
    vel[idx[0]][0] = v;
}

#[rustfmt::skip]
fn update_d2q9_bgk<const D: usize, const Q: usize>(
    even: bool,
    omega: f32,
    f: &mut VectD<VectS<f32, Q>>,
    rho: &mut VectD<f32>,
    vel: &mut VectD<VectS<f32, D>>,
    idx: VectS<i32, Q>,
) {
    // Boo; we have to make this function generic but only want D2Q9.
    assert_eq!(D, 2);
    assert_eq!(Q, 9);

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
            f[idx[0]][6],
            f[idx[0]][8],
            f[idx[0]][7],
            f[idx[0]][3],
            f[idx[0]][5],
            f[idx[0]][4],
        ]
    });

    // 0:  0  0
    // 1:  0 +1
    // 2:  0 -1
    // 3: +1  0
    // 4: +1 +1
    // 5: +1 -1
    // 6: -1  0
    // 7: -1 +1
    // 8: -1 -1

    // calc moments
    let r = f_.sum();
    let mut v: VectS<f32, D> = VectS::zero();
    v[0] = (f_[3] + f_[4] + f_[5] - f_[6] - f_[7] - f_[8]) / r;
    v[1] = (f_[1] - f_[2] + f_[4] - f_[5] + f_[7] - f_[8]) / r;
    let vv = (v * v).sum();
    let vxx = v[0] * v[0];
    let vyy = v[1] * v[1];
    let vxy = v[0] * v[1];

    // calc equilibrium & collide
    f_[0] += omega * (r * (2.0 / 9.0) * (2.0 - 3.0 * vv) - f_[0]);
    f_[1] += omega * (r * (1.0 / 18.0) * (2.0 + 6.0 * v[1] + 9.0 * vyy - 3.0 * vv) - f_[1]);
    f_[2] += omega * (r * (1.0 / 18.0) * (2.0 - 6.0 * v[1] + 9.0 * vyy - 3.0 * vv) - f_[2]);
    f_[3] += omega * (r * (1.0 / 18.0) * (2.0 + 6.0 * v[0] + 9.0 * vxx - 3.0 * vv) - f_[3]);
    f_[4] += omega * (r * (1.0 / 36.0) * (1.0 + 3.0 * (v[0] + v[1]) + 9.0 * vxy + 3.0 * vv) - f_[4]);
    f_[5] += omega * (r * (1.0 / 36.0) * (1.0 - 3.0 * (v[1] - v[0]) - 9.0 * vxy + 3.0 * vv) - f_[5]);
    f_[6] += omega * (r * (1.0 / 18.0) * (2.0 - 6.0 * v[0] + 9.0 * vxx - 3.0 * vv) - f_[6]);
    f_[7] += omega * (r * (1.0 / 36.0) * (1.0 + 3.0 * (v[1] - v[0]) - 9.0 * vxy + 3.0 * vv) - f_[7]);
    f_[8] += omega * (r * (1.0 / 36.0) * (1.0 - 3.0 * (v[0] + v[1]) + 9.0 * vxy + 3.0 * vv) - f_[8]);

    // write back to same locations
    if even {
        f[idx[0]][0] = f_[0];
        f[idx[2]][2] = f_[1];
        f[idx[1]][1] = f_[2];
        f[idx[6]][6] = f_[3];
        f[idx[8]][8] = f_[4];
        f[idx[7]][7] = f_[5];
        f[idx[3]][3] = f_[6];
        f[idx[5]][5] = f_[7];
        f[idx[4]][4] = f_[8];
    } else {
        for i in 0..9 {
            f[idx[0]][i] = f_[i];
        }
    }

    rho[idx[0]] = r;
    vel[idx[0]] = v;
}

/// Container for the LBM simulation data which is generic in the dimension and velocity set size.
pub struct LBM<const D: usize, const Q: usize> {
    ws: VectS<f32, Q>,
    qs: [VectS<f32, D>; Q],
    js: [usize; Q],

    even: bool,

    pub cnt: VectS<i32, D>,

    /// Upstream indices for each cell.
    idx: VectD<VectS<i32, Q>>,

    pub f: VectD<VectS<f32, Q>>,
    pub rho: VectD<f32>,
    pub vel: VectD<VectS<f32, D>>,
}

impl<const D: usize, const Q: usize> LBM<D, Q> {
    pub fn new(cnt: VectS<i32, D>, ws: [f32; Q], qs: [VectS<f32, D>; Q]) -> LBM<D, Q> {
        let n = cnt.prod() as usize;

        // initialize offset vectors
        let mut idx: VectD<VectS<i32, Q>> = VectD::zeros(n);
        let mut i = 0;
        for sub in raster_row_major(cnt) {
            for q in 0..Q {
                let sub_ = sub - qs[q].cast();
                let sub_ = vmod(sub_, cnt);
                let j = sub_to_idx(sub_, cnt);
                idx[i][q] = j;
            }

            i += 1;
        }

        // initialize negative indicies
        // This is a very noddy O(Q^2) and I'm sure we can do better, but Q is small.
        let mut js = [0; Q];
        for i in 0..Q {
            for j in 0..Q {
                let qij = qs[i] + qs[j];
                if (qij * qij).sum() < 1e-8 {
                    js[i] = j;
                    break;
                }
            }
        }

        // Sanity check: all indicies appear in the negative index array.
        for i in 0..Q {
            let mut found = false;
            for j in 0..Q {
                if js[j] == i {
                    found = true;
                    break;
                }
            }
            assert!(found);
        }

        let mut out = LBM {
            ws: VectS::new(ws),
            qs: qs,
            js: js,
            cnt: cnt,
            idx: idx,
            even: true,
            f: VectD::zeros(n),
            rho: VectD::ones(n),
            vel: VectD::zeros(n),
        };
        out.reinit();
        out
    }

    // Re-initialize the distribution function to the local equilibrium
    // as set by the macroscopic quantities.
    pub fn reinit(&mut self) {
        for i in 0..self.cnt.prod() {
            self.f[i] = calc_f_eq(self.rho[i], self.vel[i], self.ws, self.qs);
        }
    }

    /// Basic implementation of one iteration of the LBM method.
    pub fn step(&mut self, tau: f32) {
        let omega = 1. / tau;
        // TODO: parallelize this!
        for i in 0..self.cnt.prod() {
            if (D == 1) && (Q == 3) {
                update_d1q3_bgk(
                    self.even,
                    omega,
                    &mut self.f,
                    &mut self.rho,
                    &mut self.vel,
                    self.idx[i],
                );
            } else if (D == 2) && (Q == 9) {
                update_d2q9_bgk(
                    self.even,
                    omega,
                    &mut self.f,
                    &mut self.rho,
                    &mut self.vel,
                    self.idx[i],
                );
            } else {
                update_generic_bgk(
                    self.even,
                    omega,
                    &mut self.f,
                    &mut self.rho,
                    &mut self.vel,
                    self.idx[i],
                    self.ws,
                    self.qs,
                    self.js,
                )
            }
        }

        self.even = !self.even;
    }
}

/// Take the tensor product of two velocity sets.
///
/// #### NOTE
///
/// In Rust we cannot use expressions of const generics as const generic args themselves.
/// Thus it is necessary to have the output D & Q explicitly as arguments.
/// However, we usually don't need to specify the generic arguments because the type inference
/// works it out for us.
/// Annoyingly, if your upstream use has the wrong D & Q values this will compile,
/// but you will at least get a runtime assertion error.
pub fn tensor_prod_q<
    const D1: usize,
    const Q1: usize,
    const D2: usize,
    const Q2: usize,
    const D3: usize,
    const Q3: usize,
>(
    q1s: [VectS<f32, D1>; Q1],
    q2s: [VectS<f32, D2>; Q2],
) -> [VectS<f32, D3>; Q3] {
    // can't use const expressions as generic const args
    assert_eq!(D1 + D2, D3);
    assert_eq!(Q1 * Q2, Q3);
    let mut out = [VectS::zero(); Q3];
    let mut i = 0;
    for q1 in 0..Q1 {
        for q2 in 0..Q2 {
            for d in 0..D1 {
                out[i][d] = q1s[q1][d];
            }
            for d in 0..D2 {
                out[i][D1 + d] = q2s[q2][d];
            }
            i += 1;
        }
    }
    out
}

/// Take the tensor product of two weight sets.
///
/// #### NOTE
///
/// See the note on `tensor_prod_q` about generic const arguments.
pub fn tensor_prod_w<const Q1: usize, const Q2: usize, const Q3: usize>(
    w1s: [f32; Q1],
    w2s: [f32; Q2],
) -> [f32; Q3] {
    // can't use const expressions as generic const args
    assert_eq!(Q1 * Q2, Q3);
    let mut out = [0.0; Q3];
    let mut sum = 0.0;
    let mut i = 0;
    for q1 in 0..Q1 {
        for q2 in 0..Q2 {
            out[i] = w1s[q1] * w2s[q2];
            sum += out[i];
            i += 1;
        }
    }
    // renormalize
    for i in 0..Q3 {
        out[i] /= sum;
    }
    out
}

pub const D1Q3_W: [f32; 3] = [4. / 6., 1. / 6., 1. / 6.];
pub const D1Q3_Q: [VectS<f32, 1>; 3] = [VectS::new([0.]), VectS::new([1.0]), VectS::new([-1.0])];
