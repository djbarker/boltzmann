import numpy as np
import pytest
import tempfile

from boltzmann.core import check_lbm_params, calc_lbm_params_lu, calc_lbm_params_si, Simulation, bgk


def test_check_lbm_params():
    # Should fail because of the stability condition.
    with pytest.raises(AssertionError):
        check_lbm_params(1_000_000, 1, tau=0.51)

    # Should fail because of the Mach condition.
    with pytest.raises(AssertionError):
        check_lbm_params(1, 1, tau=1.0)

    # Should be okay
    check_lbm_params(1, 1, tau=0.51)


def test_calc_lbm_params_lu():
    tau1, u1 = calc_lbm_params_lu(1, 1)
    tau2, u2 = calc_lbm_params_lu(1, 1, tau=0.51)
    assert tau1 > tau2  # method should choose largest tau if not specified
    assert u1 > u2  # u and tau are linearly related
    assert abs(tau2 - 0.51) < 1e-8

    _, u3 = calc_lbm_params_lu(1, 1, tau=0.51)
    _, u4 = calc_lbm_params_lu(10, 1, tau=0.51)
    assert u3 < u4  # u and Re are linearly related (for fixed tau)

    _, u3 = calc_lbm_params_lu(1, 1, tau=0.51)
    _, u4 = calc_lbm_params_lu(1, 10, tau=0.51)
    assert u3 > u4  # u and L are inversely related (for fixed tau)


def test_calc_lbm_params_si():
    u = 1.0
    L = 1.0
    nu = 1e-4

    N = 1000
    Re = u * L / nu

    # free tau
    tau1, u1 = calc_lbm_params_lu(Re, N)
    tau2, u2 = calc_lbm_params_si(L / N, u, L, nu)
    assert abs(tau1 - tau2) < 1e-8
    assert abs(u1 - u2) < 1e-8

    # # fixed tau
    _, u1 = calc_lbm_params_lu(Re, N, tau=0.51)
    _, u2 = calc_lbm_params_si(L / N, u, L, nu, tau=0.51)
    assert abs(u1 - u2) < 1e-8


def test_simulation_1():
    """
    Tests construction, that the array shapes and initial values are correct, and that running works.
    """
    sim = Simulation("cpu", [100, 100], bgk(0.51))

    # Check array shapes.
    assert sim.fluid.vel.shape == (100, 100, 2)
    assert sim.fluid.rho.shape == (100, 100)
    assert sim.cells.flags.shape == (100, 100)

    # Check default values.
    assert abs(sim.fluid.vel.min() - 0) < 1e-8
    assert abs(sim.fluid.vel.max() - 0) < 1e-8
    assert abs(sim.fluid.rho.max() - 1) < 1e-8
    assert abs(sim.fluid.rho.max() - 1) < 1e-8

    # Can we run it?
    sim.iterate(10)

    # Check conservation.
    assert abs(sim.fluid.vel.mean() - 0) < 1e-4
    assert abs(sim.fluid.rho.mean() - 1) < 1e-4


def test_simulation_2():
    """
    Check adding a tracer & conservation.
    """

    sim = Simulation("cpu", [100, 100], bgk(0.51))
    tracer = sim.add_tracer("tracer", bgk(0.51))
    tracer.val[10:20, 30:40] = 1.0
    sim.iterate(10)

    # Check conservation.
    assert abs(tracer.val.sum() - 10**2) < 1e-4


def test_simulation_3():
    """
    Check adding a body force works okay.
    """
    sim = Simulation("cpu", [100, 100], bgk(1.0))
    sim.set_gravity(np.array([0, 0.1], dtype=np.float32))
    sim.iterate(10)
    assert abs(sim.fluid.vel[..., 1].mean() - 0.1 * 10) < 1e-4


def test_simulation_serde():
    """
    Check serializing & deserializing works fine.
    """

    with tempfile.TemporaryDirectory() as tmpdirname:
        path = tmpdirname + "/checkpoint.mpk"

        sim1 = Simulation("cpu", [100, 100], bgk(0.51))
        trc1 = sim1.add_tracer("tracer", bgk(0.51))
        trc1.val[10:20, 30:40] = 1.0
        sim1.add_boussinesq_coupling(trc1, 0.1, 0, np.array([0, 1], np.float32))
        sim1.iterate(10)

        # round trip
        sim1.write_checkpoint(path)
        sim2 = Simulation.load_checkpoint("cpu", path)
        trc2 = sim2.get_tracer("tracer")

        assert np.allclose(sim1.fluid.rho, sim2.fluid.rho)
        assert np.allclose(sim1.fluid.vel, sim2.fluid.vel)
        assert np.allclose(trc1.val, trc2.val)
