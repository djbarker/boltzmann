from boltzmann.utils.option import to_opt, map_opt, unwrap, Some
import pytest


def test_to_opt():
    assert to_opt(None) is None
    assert to_opt(1) == Some(1)
    assert to_opt(Some(1)) == Some(1)


def test_map_opt():
    def f(x):
        return x**2

    assert map_opt(None, f) is None
    assert map_opt(2, f) == Some(4)
    assert map_opt(Some(2), f) == Some(4)


def test_unwrap():
    with pytest.raises(TypeError):
        unwrap(None)

    assert unwrap(1) == 1
    assert unwrap(Some(1)) == 1
