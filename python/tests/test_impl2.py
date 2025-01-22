import numpy as np

from bz_numba import *
from boltzmann.core import D2Q9


def test_periodic_domain():
    cnt = np.array([100, 150])
    dom = NumbaDomain(cnt)

    # make some arrays
    X = make_array(dom, dtype=np.int32)
    Y = make_array(dom, 2)

    assert X.shape == (102 * 152,)
    assert Y.shape == (102 * 152, 2)

    # slice indexing
    ixlower = make_slice_y1d(dom, 1, 1, -1)
    iylower = make_slice_x1d(dom, 1, 1, -1)
    ixupper = make_slice_y1d(dom, -2, 1, -1)
    iyupper = make_slice_x1d(dom, -2, 1, -1)

    assert ixlower.shape == (150,)
    assert ixupper.shape == (150,)
    assert iylower.shape == (100,)
    assert iyupper.shape == (100,)

    X[ixlower] = np.array(range(150))
    X[ixupper] = np.array(range(150)) + 150
    X[iylower] = np.array(range(100)) + 300
    X[iyupper] = np.array(range(100)) + 400

    X[sub_to_idx(dom.counts, 1, 1)] = -1
    X[sub_to_idx(dom.counts, 1, -2)] = -2
    X[sub_to_idx(dom.counts, -2, -2)] = -3
    X[sub_to_idx(dom.counts, -2, 1)] = -4

    X_ = np.reshape(X, (152, 102))

    assert np.all(X_[0, :] == 0)
    assert np.all(X_[:, 0] == 0)
    assert np.all(X_[-1, :] == 0)
    assert np.all(X_[:, -1] == 0)

    # periodic copying
    dom.copy_periodic(X)

    X_ = np.reshape(X, (152, 102))

    assert np.all(X_[2:-2, -1] == np.array(range(150))[1:-1])
    assert np.all(X_[2:-2, 0] == np.array(range(150))[1:-1]) + 150
    assert np.all(X_[-1, 2:-2] == np.array(range(100))[1:-1]) + 300
    assert np.all(X_[0, 2:-2] == np.array(range(100))[1:-1]) + 400
    assert X_[0, 0] == -3
    assert X_[-1, -1] == -1
    assert X_[0, -1] == -2
    assert X_[-1, 0] == -4


def test_stream_basic():
    cnt = np.array([3, 3])
    dom = NumbaDomain(cnt)
    mdl = NumbaModel(D2Q9.ws, D2Q9.qs, D2Q9.js, dom.counts)

    wall = make_array(dom, dtype=np.int32)
    f1 = make_array(dom, 9, dtype=np.int32)
    f2 = np.zeros_like(f1)

    # unflattened views
    f1_ = unflatten(dom, f1)
    f2_ = unflatten(dom, f2)

    x = np.array(range(3))
    y = np.array(range(3))
    xx, yy = np.meshgrid(x, y)
    ii = yy * 3 + xx

    f1_[1:-1, 1:-1, :] = ii[:, :, None]

    dom.copy_periodic(f1)
    stream(f2, f1, wall, dom.counts, mdl)

    f_expected = []

    def _add_expected(*a: list[int]):
        f_expected.append(np.array(list(a)))

    # very manual test but it works
    _add_expected(
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
    )

    _add_expected(
        [2, 0, 1],
        [5, 3, 4],
        [8, 6, 7],
    )

    _add_expected(
        [6, 7, 8],
        [0, 1, 2],
        [3, 4, 5],
    )

    _add_expected(
        [1, 2, 0],
        [4, 5, 3],
        [7, 8, 6],
    )

    _add_expected(
        [3, 4, 5],
        [6, 7, 8],
        [0, 1, 2],
    )

    _add_expected(
        [8, 6, 7],
        [2, 0, 1],
        [5, 3, 4],
    )

    _add_expected(
        [7, 8, 6],
        [1, 2, 0],
        [4, 5, 3],
    )

    _add_expected(
        [4, 5, 3],
        [7, 8, 6],
        [1, 2, 0],
    )

    _add_expected(
        [5, 3, 4],
        [8, 6, 7],
        [2, 0, 1],
    )

    for i, f in enumerate(f_expected):
        try:
            assert np.all(f2_[1:-1, 1:-1, i] == f), f"{i}"
        except AssertionError:
            print(f"idx: {i}:\nsaw:\n{f2_[1:-1, 1:-1, i]}\nexpected:\n{f}")
            raise

    # Invariant: no creation or destruction of populations.
    assert sorted(f2_[1:-1, 1:-1].flatten()) == sorted(f1_[1:-1, 1:-1].flatten())


def test_stream_bounceback():
    cnt = np.array([3, 3])
    dom = NumbaDomain(cnt)
    mdl = NumbaModel(D2Q9.ws, D2Q9.qs, D2Q9.js, dom.counts)

    wall = make_array(dom, dtype=np.int32)
    f1 = make_array(dom, 9, dtype=np.int32)
    f2 = np.zeros_like(f1)

    # unflattened views
    f1_ = unflatten(dom, f1)
    f2_ = unflatten(dom, f2)

    # a wall in the centre cell
    wall[sub_to_idx(dom.counts, 2, 2)] = CellType.WALL.value

    x = np.array(range(3))
    y = np.array(range(3))
    xx, yy = np.meshgrid(x, y)
    ii = yy * 3 + xx

    # bounce back mixes between directions so give them unique values
    for i in range(9):
        f1_[1:-1, 1:-1, i] = ii[:, :] + i * 10

    dom.copy_periodic(f1)
    stream(f2, f1, wall, dom.counts, mdl)

    f_expected = []

    def _add_expected(*a: list[int]):
        f_expected.append(np.array(list(a)))

    # [0, 0]
    _add_expected(
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
    )

    # [1, 0]
    _add_expected(
        [12, 10, 11],
        [15, 14, 35],
        [18, 16, 17],
    )

    # [0, 1]
    _add_expected(
        [26, 27, 28],
        [20, 24, 22],
        [23, 47, 25],
    )

    # [-1, 0]
    _add_expected(
        [31, 32, 30],
        [13, 34, 33],
        [37, 38, 36],
    )

    # [0, -1]
    _add_expected(
        [43, 21, 45],
        [46, 44, 48],
        [40, 41, 42],
    )

    # [1, 1]
    _add_expected(
        [58, 56, 57],
        [52, 54, 51],
        [55, 53, 78],
    )

    # [-1, 1]
    _add_expected(
        [67, 68, 66],
        [61, 64, 60],
        [86, 65, 63],
    )

    # [-1, -1]
    _add_expected(
        [50, 75, 73],
        [77, 74, 76],
        [71, 72, 70],
    )

    # [1, -1]
    _add_expected(
        [85, 83, 62],
        [88, 84, 87],
        [82, 80, 81],
    )

    for i, f in enumerate(f_expected):
        try:
            assert np.all(f2_[1:-1, 1:-1, i] == f), f"{i}"
        except AssertionError:
            print(f"idx: {i}:\nsaw:\n{f2_[1:-1, 1:-1, i]}\nexpected:\n{f}")
            raise

    # Invariant: no creation or destruction of populations.
    assert sorted(f2_[1:-1, 1:-1].flatten()) == sorted(f1_[1:-1, 1:-1].flatten())


def test_calc_equilibrium():
    cnt = np.array([3, 3])
    dom = NumbaDomain(cnt)
    mdl = NumbaModel(D2Q9.ws, D2Q9.qs, D2Q9.js, dom.counts)
    cs = 100.0

    r = make_array(dom, fill=1)
    v = make_array(dom, 2)
    f = make_array(dom, 9)

    calc_equilibrium(v, r, f, np.float32(cs), mdl)


def test_loop():
    cnt = np.array([3, 3])
    dom = NumbaDomain(cnt)
    mdl = NumbaModel(D2Q9.ws, D2Q9.qs, D2Q9.js, dom.counts)
    prm = NumbaParams(0.01, 0.1, 10, 0.6)  # made up params

    r = make_array(dom, fill=1)
    c = make_array(dom)
    v = make_array(dom, 2)
    f = make_array(dom, 9)
    feq = make_array(dom, 9)
    is_wall = make_array(dom, dtype=np.int32)
    update_vel = make_array(dom, dtype=np.int32)

    assert np.max(f) < 1e-8

    loop_for(10, v, r, c, f, feq, is_wall, update_vel, prm, dom, mdl)

    assert np.min(f) > 1e-8
