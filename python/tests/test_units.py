from boltzmann.units import Scales, UnitConverter, Domain
import numpy as np


def test_unit_converter():
    u = UnitConverter(0.01)
    assert abs(u.to_physical_units(1) - 0.01) < 1e-8
    assert abs(u.to_lattice_units(1) - 100) < 1e-8


def test_scales():
    dx = 0.01
    dt = 0.0025
    s = Scales(dx=dx, dt=dt)

    assert abs(s.time.to_lattice_units(dt) - 1) < 1e-8
    assert abs(s.distance.to_lattice_units(dx) - 1) < 1e-8
    assert abs(s.velocity.to_lattice_units(dx / dt) - 1) < 1e-8
    assert abs(s.acceleration.to_lattice_units(dx / dt**2) - 1) < 1e-8


def test_domain_make():
    # Various combinations which should all be equivalent
    dx = 0.01
    L = 1.0
    N = int(L / dx)

    dom1 = Domain.make(upper=[L, L], dx=dx)
    dom2 = Domain.make(counts=[N, N], dx=dx)
    dom3 = Domain.make(upper=[L, L], counts=[N, N])
    dom4 = Domain.make(upper=[L, L], counts=[N, N], dx=dx)

    def _compare(d1: Domain, d2: Domain):
        assert (d1.counts == d2.counts).all()
        assert abs(d1.dx - d2.dx) < 1e-8
        assert np.allclose(d1.lower, d2.lower)
        assert np.allclose(d1.upper, d2.upper)

    _compare(dom1, dom2)
    _compare(dom1, dom3)
    _compare(dom1, dom4)
