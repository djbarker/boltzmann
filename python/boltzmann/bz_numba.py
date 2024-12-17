"""
An implementation which uses numba.
"""

from __future__ import annotations

import numpy as np
import numba
from numba.experimental import jitclass
from numba import void, int32, int64, float32

from boltzmann.core import CellType, DimsT, to_dims, D2Q9 as D2Q9_, D2Q5 as D2Q5_


def jc_arg(cls: type):
    """
    Use a @jitclass type as an argument in eager @jit signature.
    """
    return cls.class_type.instance_type


@jitclass(
    spec={
        "ws": numba.float32[:],
        "qs": numba.int32[:, ::1],
        "js": numba.int32[:],
        "qs_f32": numba.float32[:, ::1],
    }
)  # type: ignore
class NumbaModel:
    def __init__(self, ws: np.ndarray, qs: np.ndarray, js: np.ndarray):
        self.ws = ws
        self.qs = qs
        self.js = js
        self.qs_f32 = qs.astype(np.float32).copy()


D2Q9 = NumbaModel(D2Q9_.ws, D2Q9_.qs, D2Q9_.js)
D2Q5 = NumbaModel(D2Q5_.ws, D2Q5_.qs, D2Q5_.js)


@jitclass(
    spec={
        "dt_si": float32,
        "dx_si": float32,
        "w_pos_lu": float32,
        "w_neg_lu": float32,
        "g_lu": float32[:],
    }
)  # type: ignore
class NumbaParams:
    def __init__(
        self,
        dt_si: float,
        dx_si: float,
        w_pos_lu: float,
        w_neg_lu: float,
        g_lu: np.ndarray,
    ):
        self.dt_si = dt_si
        self.dx_si = dx_si
        self.w_pos_lu = w_pos_lu
        self.w_neg_lu = w_neg_lu
        self.g_lu = g_lu


@jitclass(
    spec={
        "counts": numba.int32[:],
        "dims": numba.int32,
    }
)  # type: ignore
class NumbaDomain:
    def __init__(self, counts: np.ndarray):
        self.counts = (counts + 2).astype(np.int32)
        self.dims = len(counts)

    def copy_periodic(self, arr: np.ndarray):
        copy_periodic(self.counts, arr)

    def sub_to_idx(self, xidx: int, yidx: int) -> int:
        return sub_to_idx(self.counts, xidx, yidx)


@numba.njit(
    [
        void(int32[:], float32[:, ::1]),
        void(int32[:], float32[::1]),
        void(int32[:], int32[:, ::1]),
        void(int32[:], int32[::1]),
    ],
    parallel=True,
)
def copy_periodic(counts: np.ndarray, arr: np.ndarray) -> None:
    assert np.prod(counts) == arr.shape[0]

    for yidx in numba.prange(counts[1]):
        # xlower -> xupper:
        arr[yidx * counts[0] + (counts[0] - 1)] = arr[yidx * counts[0] + 1]
        # xupper -> xlower:
        arr[yidx * counts[0] + 0] = arr[yidx * counts[0] + (counts[0] - 2)]

    for xidx in numba.prange(counts[0]):
        # ylower -> yupper:
        arr[(counts[1] - 1) * counts[0] + xidx] = arr[1 * counts[0] + xidx]
        # yupper -> ylower:
        arr[0 * counts[0] + xidx] = arr[(counts[1] - 2) * counts[0] + xidx]


@numba.njit(
    int32(int32[:], int32, int32),
)
def sub_to_idx(counts: np.ndarray, xidx: int, yidx: int) -> int:
    if xidx < 0:
        xidx = counts[0] + xidx
    if yidx < 0:
        yidx = counts[1] + yidx
    return yidx * counts[0] + xidx


@numba.njit(
    [
        void(float32[:, ::1], float32[:, ::1], int32[:], int32[:], jc_arg(NumbaModel)),
        void(int32[:, ::1], int32[:, ::1], int32[:], int32[:], jc_arg(NumbaModel)),
    ],
    parallel=True,
)
def stream(
    f_to: np.ndarray,
    f_from: np.ndarray,
    is_wall: np.ndarray,
    counts: np.ndarray,
    model: NumbaModel,
):
    assert np.prod(counts) == f_to.shape[0]
    assert np.prod(counts) == f_from.shape[0]

    for yidx in numba.prange(1, counts[1] - 1):
        # for yidx in range(1, counts[1] - 1):
        for xidx in range(1, counts[0] - 1):
            idx = sub_to_idx(counts, xidx, yidx)
            for i in range(9):
                q = model.qs[i]
                j = model.js[i]
                idx_up = sub_to_idx(counts, xidx - q[0], yidx - q[1])
                # fmt: off
                f_to[idx, i] = (
                    (
                          f_from[idx_up, i] * (1 - is_wall[idx_up])  # stream
                        + f_from[idx,    j] * (0 + is_wall[idx_up])  # bounceback
                                   ) * (1 - is_wall[idx]) 
                    + f_from[idx, i] * (0 + is_wall[idx])
                )
                # fmt: on


@numba.njit(
    void(
        int32,
        float32[:, ::1],
        float32,
        float32[:],
        float32,
        float32,
    ),
    inline="always",
    fastmath=True,
)
def collide_bgk_d2q9(
    idx: int,
    f_to: np.ndarray,
    rho: float,
    v: np.ndarray,
    vv: float,
    omega: float,
):
    # Already D2Q9 specific so write it out
    # Writing it out explicitly like this does seem a bit faster.
    vxy = v[0] * v[1]
    # fmt: off
    f_to[idx, 0] -= omega * (f_to[idx, 0] - rho * (2/9)  * (2 - 3 * vv))
    f_to[idx, 1] -= omega * (f_to[idx, 1] - rho * (1/18) * (2 + 6 * v[0] + 9 * v[0]**2 - 3 * vv))
    f_to[idx, 2] -= omega * (f_to[idx, 2] - rho * (1/18) * (2 - 6 * v[0] + 9 * v[0]**2 - 3 * vv))
    f_to[idx, 3] -= omega * (f_to[idx, 3] - rho * (1/18) * (2 + 6 * v[1] + 9 * v[1]**2 - 3 * vv))
    f_to[idx, 4] -= omega * (f_to[idx, 4] - rho * (1/18) * (2 - 6 * v[1] + 9 * v[1]**2 - 3 * vv))
    f_to[idx, 5] -= omega * (f_to[idx, 5] - rho * (1/36) * (1 + 3 * (v[0] + v[1]) + 9 * vxy + 3 * vv))
    f_to[idx, 6] -= omega * (f_to[idx, 6] - rho * (1/36) * (1 - 3 * (v[0] + v[1]) + 9 * vxy + 3 * vv))
    f_to[idx, 7] -= omega * (f_to[idx, 7] - rho * (1/36) * (1 + 3 * (v[1] - v[0]) - 9 * vxy + 3 * vv))
    f_to[idx, 8] -= omega * (f_to[idx, 8] - rho * (1/36) * (1 - 3 * (v[1] - v[0]) - 9 * vxy + 3 * vv))
    # fmt: on


@numba.njit(
    void(
        int32,
        float32[:, ::1],
        float32,
        float32[:],
        float32,
    ),
    inline="always",
    fastmath=True,
)
def collide_bgk_d2q5_advdif(
    idx: int,
    f_to: np.ndarray,
    C: float,
    v: np.ndarray,
    omega: float,
):
    vv = np.sum(v * v)
    # fmt: off
    f_to[idx, 0] -= omega * (f_to[idx, 0] - C * (1/6) * (2 - 3 * vv))
    f_to[idx, 1] -= omega * (f_to[idx, 1] - C * (1/12) * (2 + 6 * v[0] + 9 * v[0]**2 - 3 * vv))
    f_to[idx, 2] -= omega * (f_to[idx, 2] - C * (1/12) * (2 - 6 * v[0] + 9 * v[0]**2 - 3 * vv))
    f_to[idx, 3] -= omega * (f_to[idx, 3] - C * (1/12) * (2 + 6 * v[1] + 9 * v[1]**2 - 3 * vv))
    f_to[idx, 4] -= omega * (f_to[idx, 4] - C * (1/12) * (2 - 6 * v[1] + 9 * v[1]**2 - 3 * vv))
    # fmt: on


@numba.njit(
    void(
        int32,
        float32[:, ::1],
        float32,
        float32[:],
        float32,
        jc_arg(NumbaParams),
        jc_arg(NumbaModel),
    ),
    inline="always",
    fastmath=True,
)
def collide_trt_d2q9(
    idx: int,
    f_to: np.ndarray,
    rho_si: float,
    v_lu: np.ndarray,
    vv_lu: float,
    params: NumbaParams,
    model: NumbaModel,
):
    feq = np.zeros((9,))
    f_ = np.zeros((9,))
    feq_ = np.zeros((9,))

    # equilibrate
    for i in range(9):
        w = model.ws[i]
        q_lu = model.qs[i]
        qv_lu = np.sum(q_lu * v_lu)

        feq[i] = rho_si * w * (1 + 3.0 * qv_lu + 4.5 * qv_lu**2 - (3.0 / 2.0) * vv_lu)

    f_[0] = f_to[idx, 0]
    f_[1] = (f_to[idx, 1] + f_to[idx, 2]) * 0.5
    f_[2] = (f_to[idx, 1] - f_to[idx, 2]) * 0.5
    f_[3] = (f_to[idx, 3] + f_to[idx, 4]) * 0.5
    f_[4] = (f_to[idx, 3] - f_to[idx, 4]) * 0.5
    f_[5] = (f_to[idx, 5] + f_to[idx, 6]) * 0.5
    f_[6] = (f_to[idx, 5] - f_to[idx, 6]) * 0.5
    f_[7] = (f_to[idx, 7] + f_to[idx, 8]) * 0.5
    f_[8] = (f_to[idx, 7] - f_to[idx, 8]) * 0.5

    feq_[0] = feq[0]
    feq_[1] = (feq[1] + feq[2]) * 0.5
    feq_[2] = (feq[1] - feq[2]) * 0.5
    feq_[3] = (feq[3] + feq[4]) * 0.5
    feq_[4] = (feq[3] - feq[4]) * 0.5
    feq_[5] = (feq[5] + feq[6]) * 0.5
    feq_[6] = (feq[5] - feq[6]) * 0.5
    feq_[7] = (feq[7] + feq[8]) * 0.5
    feq_[8] = (feq[7] - feq[8]) * 0.5

    f_to[idx, 0] -= params.w_pos_lu * (f_[0] - feq_[0])
    f_to[idx, 1] -= params.w_pos_lu * (f_[1] - feq_[1]) + params.w_neg_lu * (f_[2] - feq_[2])
    f_to[idx, 2] -= params.w_pos_lu * (f_[1] - feq_[1]) - params.w_neg_lu * (f_[2] - feq_[2])
    f_to[idx, 3] -= params.w_pos_lu * (f_[3] - feq_[3]) + params.w_neg_lu * (f_[4] - feq_[4])
    f_to[idx, 4] -= params.w_pos_lu * (f_[3] - feq_[3]) - params.w_neg_lu * (f_[4] - feq_[4])
    f_to[idx, 5] -= params.w_pos_lu * (f_[5] - feq_[5]) + params.w_neg_lu * (f_[6] - feq_[6])
    f_to[idx, 6] -= params.w_pos_lu * (f_[5] - feq_[5]) - params.w_neg_lu * (f_[6] - feq_[6])
    f_to[idx, 7] -= params.w_pos_lu * (f_[7] - feq_[7]) + params.w_neg_lu * (f_[8] - feq_[8])
    f_to[idx, 8] -= params.w_pos_lu * (f_[7] - feq_[7]) - params.w_neg_lu * (f_[8] - feq_[8])


@numba.njit(
    [
        void(
            float32[:, ::1],
            float32[:, ::1],
            int32[:],
            int32[:],
            int32[:],
            jc_arg(NumbaParams),
            jc_arg(NumbaModel),
        ),
    ],
    parallel=True,
    fastmath=True,
)
def stream_and_collide(
    f_to: np.ndarray,
    f_from: np.ndarray,
    is_wall: np.ndarray,
    is_fixed: np.ndarray,
    counts: np.ndarray,
    params: NumbaParams,
    model: NumbaModel,
):
    assert f_to is not f_from
    assert np.prod(counts) == f_to.shape[0]
    assert np.prod(counts) == f_from.shape[0]

    qs = np.ascontiguousarray(model.qs_f32.T)

    for yidx in numba.prange(1, counts[1] - 1):
        for xidx in range(1, counts[0] - 1):
            idx = sub_to_idx(counts, xidx, yidx)

            # fmt: off
            update_f = ( 1 
                 * (1 - is_wall[idx])  # not wall
                 * (1- is_fixed[idx])  # not fixed density & velocity / concentration
            )
            # fmt: on

            if update_f < 1:
                continue

            # stream
            f_to[idx, 0] = f_from[idx, 0]
            for i in range(1, len(model.qs)):
                q = model.qs[i]
                j = model.js[i]
                idx_up = sub_to_idx(counts, xidx - q[0], yidx - q[1])
                # fmt: off
                f_to[idx, i] = (
                    (
                          f_from[idx_up, i] * (1 - is_wall[idx_up])  # stream
                        + f_from[idx,    j] * (0 + is_wall[idx_up])  # bounceback
                                   )
                )
                # fmt: on

            # macroscopic
            rho = np.sum(f_to[idx, :])
            v_lu = np.dot(qs, f_to[idx, :] / rho)
            v_lu += params.g_lu / params.w_pos_lu

            vv_lu = np.sum(v_lu * v_lu)

            collide_bgk_d2q9(idx, f_to, rho, v_lu, vv_lu, params.w_pos_lu)
            # collide_trt_d2q9(idx, f_to, rho_si, v_lu, vv_lu, params, model)


@numba.njit(
    void(
        int64,
        float32[:, ::1],
        float32[:],
        float32[:, ::1],
        float32[:, ::1],
        int32[:],
        int32[:],
        jc_arg(NumbaParams),
        jc_arg(NumbaDomain),
        jc_arg(NumbaModel),
    ),
    parallel=True,
)
def loop_for_2(
    iters: int,
    vel,  # in lattice-units
    rho,
    f1_si,
    f2_si,
    is_wall,
    is_fixed,
    params: NumbaParams,
    pidx: NumbaDomain,
    model: NumbaModel,
):
    counts = pidx.counts

    assert f1_si is not f2_si
    assert np.prod(counts) == f1_si.shape[0]
    assert np.prod(counts) == f2_si.shape[0]
    assert np.prod(counts) == vel.shape[0]
    assert np.prod(counts) == rho.shape[0]

    for i in range(iters):
        if i % 2 == 0:
            pidx.copy_periodic(f1_si)
            stream_and_collide(f2_si, f1_si, is_wall, is_fixed, counts, params, model)
        else:
            pidx.copy_periodic(f2_si)
            stream_and_collide(f1_si, f2_si, is_wall, is_fixed, counts, params, model)

    # make sure output always ends in f1
    if iters % 2 != 0:
        f1_si[:] = f2_si[:]

    # output macroscopic
    qs = np.ascontiguousarray(model.qs_f32.T)
    for yidx in numba.prange(1, counts[1] - 1):
        for xidx in range(1, counts[0] - 1):
            idx = sub_to_idx(counts, xidx, yidx)

            rho_ = np.sum(f1_si[idx, :])
            vel_ = np.dot(qs, f1_si[idx, :] / rho_)

            vel[idx, :] = vel_
            rho[idx] = rho_


def calc_curl_2d(pidx: NumbaDomain, v: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """
    All quantities in lattice-units
    """
    counts = pidx.counts

    # calc curl
    pidx.copy_periodic(v)
    curl = np.zeros((v.shape[0],))
    for yidx in range(1, counts[1] - 1):
        for xidx in range(1, counts[0] - 1):
            idx = yidx * counts[0] + xidx

            # NOTE: Assumes zero wall velocity.
            # fmt: off
            dydx1 = v[idx - counts[0], 0] * (cell[idx - counts[0]] != CellType.BC_WALL.value)
            dydx2 = v[idx + counts[0], 0] * (cell[idx + counts[0]] != CellType.BC_WALL.value)
            dxdy1 = v[idx -         1, 1] * (cell[idx -         1] != CellType.BC_WALL.value)
            dxdy2 = v[idx +         1, 1] * (cell[idx +         1] != CellType.BC_WALL.value)
            # fmt: on

            curl[idx] = ((dydx2 - dydx1) - (dxdy2 - dxdy1)) / 2

    return curl


@numba.njit(
    [
        void(
            float32[:, ::1],
            float32[:, ::1],
            float32[:, ::1],
            int32[:],
            int32[:],
            int32[:],
            jc_arg(NumbaParams),
            jc_arg(NumbaModel),
        ),
    ],
    parallel=True,
    fastmath=True,
)
def stream_and_collide_advdif(
    vel: np.ndarray,  # in lattice units
    f_to: np.ndarray,
    f_from: np.ndarray,
    is_wall: np.ndarray,
    is_fixed: np.ndarray,
    counts: np.ndarray,
    params: NumbaParams,
    model: NumbaModel,
):
    """
    TODO: This is basically a copy-paste of `stream_and_collide` but adapted for advection-diffusion.
          They are very similar and should be combined.
    """
    assert f_to is not f_from
    assert np.prod(counts) == f_to.shape[0]
    assert np.prod(counts) == f_from.shape[0]

    # PONDER: this loop is being duplicated accross the two calls to stream_and_collide (one for
    #         the fluid and one for adv. dif.) but perhaps combining is worse due to harming memory
    #         locality?

    for yidx in numba.prange(1, counts[1] - 1):
        for xidx in range(1, counts[0] - 1):
            idx = sub_to_idx(counts, xidx, yidx)

            # fmt: off
            update_f = ( 1 
                 * (1 - is_wall[idx])  # not wall
                 * (1- is_fixed[idx])  # not fixed density & velocity / concentration
            )
            # fmt: on

            if update_f < 1:
                continue

            # stream
            f_to[idx, 0] = f_from[idx, 0]
            for i in range(1, len(model.qs)):
                q = model.qs[i]
                j = model.js[i]
                idx_up = sub_to_idx(counts, xidx - q[0], yidx - q[1])
                # fmt: off
                f_to[idx, i] = (
                    (
                          f_from[idx_up, i] * (1 - is_wall[idx_up])  # stream
                        + f_from[idx,    j] * (0 + is_wall[idx_up])  # bounceback
                                   )
                )
                # fmt: on

            # macroscopic
            C = np.sum(f_to[idx, :])
            v_lu = vel[idx, :]

            collide_bgk_d2q5_advdif(idx, f_to, C, v_lu, params.w_pos_lu)


@numba.njit(
    void(
        int64,
        float32[:],
        float32[:, ::1],
        float32[:, ::1],
        float32[:, ::1],
        jc_arg(NumbaModel),
        float32[:],
        float32[:, ::1],
        float32[:, ::1],
        jc_arg(NumbaModel),
        int32[:],
        int32[:],
        jc_arg(NumbaParams),
        jc_arg(NumbaDomain),
    ),
    parallel=True,
)
def loop_for_2_advdif(
    iters: int,
    rho,
    vel,  # in lattice-units
    f1,
    f2,
    model_f: NumbaModel,
    conc,
    g1,
    g2,
    model_g: NumbaModel,
    is_wall,
    is_fixed,
    params: NumbaParams,
    pidx: NumbaDomain,
):
    counts = pidx.counts

    assert g1 is not g2
    assert np.prod(counts) == g1.shape[0]
    assert np.prod(counts) == g2.shape[0]
    assert np.prod(counts) == vel.shape[0]
    assert np.prod(counts) == conc.shape[0]

    for i in range(iters):
        if i % 2 == 0:
            pidx.copy_periodic(f1)
            stream_and_collide(f2, f1, is_wall, is_fixed, counts, params, model_f)

            pidx.copy_periodic(g1)
            stream_and_collide_advdif(vel, g2, g1, is_wall, is_fixed, counts, params, model_g)
        else:
            pidx.copy_periodic(f2)
            stream_and_collide(f1, f2, is_wall, is_fixed, counts, params, model_f)

            pidx.copy_periodic(g2)
            stream_and_collide_advdif(vel, g1, g2, is_wall, is_fixed, counts, params, model_g)

    # make sure output always ends in f1
    if iters % 2 != 0:
        f1[:] = f2[:]
        g1[:] = g2[:]

    # output macroscopic
    qs = np.ascontiguousarray(model_f.qs_f32.T)
    for yidx in numba.prange(1, counts[1] - 1):
        for xidx in range(1, counts[0] - 1):
            idx = sub_to_idx(counts, xidx, yidx)

            rho_ = np.sum(f1[idx, :])
            vel_ = np.dot(qs, f1[idx, :] / rho_)
            conc_ = np.sum(g1[idx, :])

            vel[idx, :] = vel_
            rho[idx] = rho_
            conc[idx] = conc_
