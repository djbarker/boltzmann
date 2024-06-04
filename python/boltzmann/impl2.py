"""
An implementation which uses numba.
"""

import numpy as np

from typing import Type

import numba
from numba.experimental import jitclass
from numba import void, int32, int64, float32, float64

from boltzmann.core import CellType, DimsT, to_dims


def jc_arg(cls: Type):
    """
    Use a @jitclass type as an argument in eager @jit signature.
    """
    return cls.class_type.instance_type


@jitclass(
    {
        "ws": numba.float32[:],
        "qs": numba.int32[:, :],
        "js": numba.int32[:],
        "os": numba.int32[:],
        "qs_f32": numba.float32[:, ::1],
    }
)
class NumbaModel:
    def __init__(self, ws: np.ndarray, qs: np.ndarray, js: np.ndarray, counts: np.ndarray):
        self.ws = ws
        self.qs = qs
        self.js = js
        self.os = qs[:, 0] + qs[:, 1] * counts[0]  # downstream offset
        self.qs_f32 = qs.astype(np.float32).copy()


@jitclass(
    {
        "dt": float32,
        "dx": float32,
        "cs": float32,
        "tau": float32,
        "wpve": float32,
        "wnve": float32,
    }
)
class NumbaParams:
    def __init__(
        self,
        dt: float,
        dx: float,
        cs: float,
        tau: float,
        wpve: float,
        wnve: float,
    ):
        self.dt = dt
        self.dx = dx
        self.cs = cs
        self.tau = tau
        self.wpve = wpve
        self.wnve = wnve


@jitclass(
    {
        "counts": numba.int32[:],
        "dims": numba.int32,
    }
)
class PeriodicDomain:
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


def make_array(
    dom: PeriodicDomain,
    dims: DimsT = None,
    dtype: np.dtype = np.float32,
    fill: float | int = 0.0,
) -> np.ndarray:
    """
    Make a (possibly multi-dimensional) array where the first dimension is the 1d lattice index.
    """
    dims = to_dims(dims)
    dims = [np.prod(dom.counts)] + dims
    return np.full(dims, fill, dtype)


def flatten(dom: PeriodicDomain, arr: np.ndarray) -> np.ndarray:
    assert np.ndim(arr) >= dom.dims
    return np.reshape(arr, (np.prod(dom.counts),) + arr.shape[dom.dims :])


def unflatten(dom: PeriodicDomain, arr: np.ndarray, rev: bool = True) -> np.ndarray:
    n = arr.shape[1:]
    c = tuple(dom.counts)
    if rev:
        c = c[::-1]
    return np.reshape(arr, c + n)


def make_slice_x1d(dom: PeriodicDomain, yidx: int, start: int = 0, stop: int = -1) -> np.ndarray:
    if stop < 0:
        stop = dom.counts[0] + stop
    if yidx < 0:
        yidx = dom.counts[1] + yidx
    xidx = np.arange(start=start, stop=stop, step=1, dtype=np.int32)
    return (yidx * dom.counts[0] + xidx).astype(np.int32)


def make_slice_y1d(dom: PeriodicDomain, xidx: int, start: int = 0, stop: int = -1) -> np.ndarray:
    if stop < 0:
        stop = dom.counts[1] + stop
    if xidx < 0:
        xidx = dom.counts[0] + xidx
    yidx = np.arange(start=start, stop=stop, step=1, dtype=np.int32)
    return (yidx * dom.counts[0] + xidx).astype(np.int32)


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
    void(float32[:, ::1], float32[:], float32[:, ::1], float32, jc_arg(NumbaModel)),
    parallel=True,
)
def calc_equilibrium(
    v: np.ndarray,
    rho: np.ndarray,
    feq: np.ndarray,
    cs: float,
    model: NumbaModel,
) -> np.ndarray:
    for idx in numba.prange(v.shape[0]):
        vv = np.sum(v[idx, :] ** 2)
        for i in range(9):
            w = model.ws[i]
            q = model.qs[i]
            qv = np.sum(q * v[idx, :])
            feq[idx, i] = (
                rho[idx]
                * w
                * (1 + 3.0 * qv / cs**1 + 4.5 * qv**2 / cs**2 - (3.0 / 2.0) * vv / cs**2)
            )

    # return feq


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
        int64,
        float32[:, ::1],
        float32[:],
        float32[:, ::1],
        float32[:, ::1],
        int32[:],
        int32[:],
        jc_arg(NumbaParams),
        jc_arg(PeriodicDomain),
        jc_arg(NumbaModel),
    ),
    parallel=True,
)
def loop_for(
    iters: int,
    v,
    rho,
    f,
    feq,
    is_wall,
    update_vel,
    params: NumbaParams,
    pidx: PeriodicDomain,
    model: NumbaModel,
):

    cs = params.cs
    counts = pidx.counts

    assert f is not feq
    assert np.prod(counts) == f.shape[0]
    assert np.prod(counts) == feq.shape[0]
    assert np.prod(counts) == v.shape[0]
    assert np.prod(counts) == rho.shape[0]

    for _ in range(iters):

        # equillibrium
        calc_equilibrium(v, rho, feq, cs, model)

        if np.any(~np.isfinite(feq)):
            raise ValueError(f"non finite value in feq")

        # collide
        f += (feq - f) / params.tau

        # periodic
        copy_periodic(counts, f)

        # stream
        stream(feq, f, is_wall, counts, model)

        # swap
        # f, feq = feq, f
        # TODO: not sure why but just swapping vars causes issues with numba
        f[:] = feq[:]

        # macroscopic
        rho[:] = np.sum(f, -1)
        v_ = f @ model.qs_f32

        # numba parallel does not like broadcast... sad.
        # v[:] = update_vel[:, None] * v_ * cs / rho[:, None] + (1 - update_vel[:, None]) * v
        for vdim in numba.prange(2):
            v[:, vdim] = update_vel * v_[:, vdim] * cs / rho + (1 - update_vel) * v[:, vdim]


@numba.njit(
    void(
        int32,
        float32[:, :],
        float32,
        float32[:],
        float32,
        jc_arg(NumbaParams),
        jc_arg(NumbaModel),
    ),
    inline="always",
)
def collide_bgk_d2q9(
    idx: int,
    f_to: np.ndarray,
    rho: float,
    v: np.ndarray,
    vv: float,
    params: NumbaParams,
    model: NumbaModel,
):

    for i in range(9):
        w = model.ws[i]
        q = model.qs[i]
        qv = np.sum(q * v)

        feq = rho * w * (1 + 3.0 * qv + 4.5 * qv**2 - (3.0 / 2.0) * vv)

        f_to[idx, i] = f_to[idx, i] + (feq - f_to[idx, i]) / params.tau


@numba.njit(
    void(
        int32,
        float32[:, :],
        float32,
        float32[:],
        float32,
        jc_arg(NumbaParams),
        jc_arg(NumbaModel),
    ),
    inline="always",
)
def collide_trt_d2q9(
    idx: int,
    f_to: np.ndarray,
    rho: float,
    v: np.ndarray,
    vv: float,
    params: NumbaParams,
    model: NumbaModel,
):
    feq = np.zeros((9,))
    f_ = np.zeros((9,))
    feq_ = np.zeros((9,))

    # equilibrate
    for i in range(9):
        w = model.ws[i]
        q = model.qs[i]
        qv = np.sum(q * v)

        feq[i] = rho * w * (1 + 3.0 * qv + 4.5 * qv**2 - (3.0 / 2.0) * vv)

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

    # fmt: off
    f_to[idx, 0] -= (params.wpve * (f_[0] - feq_[0]))
    f_to[idx, 1] -= (params.wpve * (f_[1] - feq_[1]) + params.wnve * (f_[2] - feq_[2]))
    f_to[idx, 2] -= (params.wpve * (f_[1] - feq_[1]) - params.wnve * (f_[2] - feq_[2]))
    f_to[idx, 3] -= (params.wpve * (f_[3] - feq_[3]) + params.wnve * (f_[4] - feq_[4]))
    f_to[idx, 4] -= (params.wpve * (f_[3] - feq_[3]) - params.wnve * (f_[4] - feq_[4]))
    f_to[idx, 5] -= (params.wpve * (f_[5] - feq_[5]) + params.wnve * (f_[6] - feq_[6]))
    f_to[idx, 6] -= (params.wpve * (f_[5] - feq_[5]) - params.wnve * (f_[6] - feq_[6]))
    f_to[idx, 7] -= (params.wpve * (f_[7] - feq_[7]) + params.wnve * (f_[8] - feq_[8]))
    f_to[idx, 8] -= (params.wpve * (f_[7] - feq_[7]) - params.wnve * (f_[8] - feq_[8]))
    # fmt: on


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
            float32[:, :],
            float32[:],
        ),
    ],
    parallel=True,
)
def stream_and_collide(
    f_to: np.ndarray,
    f_from: np.ndarray,
    is_wall: np.ndarray,
    update_vel: np.ndarray,
    counts: np.ndarray,
    params: NumbaParams,
    model: NumbaModel,
    v_o: np.ndarray,
    rho_o: np.ndarray,
):
    assert f_to is not f_from
    assert np.prod(counts) == f_to.shape[0]
    assert np.prod(counts) == f_from.shape[0]

    cs = params.cs
    qs = np.ascontiguousarray(model.qs_f32.T)

    for yidx in numba.prange(1, counts[1] - 1):
        for xidx in range(1, counts[0] - 1):
            idx = sub_to_idx(counts, xidx, yidx)

            # fmt: off
            update_f = ( 1 
                 * (1 - is_wall[idx])  # not wall
                 * update_vel[idx]  # not fixed velocity
            )
            # fmt: on

            if update_f < 1:
                continue

            # stream
            f_to[idx, 0] = f_from[idx, 0]
            for i in range(1, 9):
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
            v = np.dot(qs, f_to[idx, :]) / rho
            vv = np.sum(v * v)

            v_o[idx, :] = v * cs
            rho_o[idx] = rho

            # BGK
            # for i in range(9):
            #     w = model.ws[i]
            #     q = model.qs[i]
            #     qv = np.sum(q * v)

            #     feq = rho * w * (1 + 3.0 * qv + 4.5 * qv**2 - (3.0 / 2.0) * vv)

            #     f_to[idx, i] = f_to[idx, i] + (feq - f_to[idx, i]) / params.tau

            # collide_bgk_d2q9(idx, f_to, rho, v, vv, params, model)
            collide_trt_d2q9(idx, f_to, rho, v, vv, params, model)


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
        jc_arg(PeriodicDomain),
        jc_arg(NumbaModel),
    ),
    parallel=True,
)
def loop_for_2(
    iters: int,
    v,
    rho,
    f1,
    f2,
    is_wall,
    update_vel,
    params: NumbaParams,
    pidx: PeriodicDomain,
    model: NumbaModel,
):

    counts = pidx.counts

    assert f1 is not f2
    assert np.prod(counts) == f1.shape[0]
    assert np.prod(counts) == f2.shape[0]
    assert np.prod(counts) == v.shape[0]
    assert np.prod(counts) == rho.shape[0]

    for i in range(iters):

        if i % 2 == 0:
            pidx.copy_periodic(f1)
            stream_and_collide(f2, f1, is_wall, update_vel, counts, params, model, v, rho)
        else:
            pidx.copy_periodic(f2)
            stream_and_collide(f1, f2, is_wall, update_vel, counts, params, model, v, rho)

    # make sure output always ends in f1
    if iters % 2 != 0:
        f1[:] = f2[:]


def calc_curl_2d(pidx: PeriodicDomain, v: np.ndarray, cell: np.ndarray, dx: float) -> np.ndarray:
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

            curl[idx] = ((dydx2 - dydx1) - (dxdy2 - dxdy1)) / (2 * dx)

    return curl
