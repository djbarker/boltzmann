import numpy as np

from boltzmann.units import Scales

__all__ = [
    "calc_vmag",
    "calc_curl",
    "calc_stream_func",
]


def calc_vmag(val: np.ndarray, scales: Scales | None = None) -> np.ndarray:
    """
    Calculate the magnitude of a 2D or 3D vector field.

    Args:
        val: The (velocty) vector field.
        scales: Optional :py:class:`Scales` object.
                If provided, the array will be rescaled from lattice to physical units.

    Returns:
        The (scalar) magnitude of the vector field.
    """

    out = np.sqrt(np.sum(val**2, axis=-1))

    if scales is not None:
        out[:] = scales.velocity.to_physical_units(out)

    return out


def calc_curl(val: np.ndarray, scales: Scales | None = None) -> np.ndarray:
    """
    Calculate the curl of a 2D or 3D vector field.

    If the input vector field is 2D this will return a scalar array with the same *x* and *y* dimensions.
    If the input vector field is 3D the returned array will be the same shape as the input.

    Note:
        Does not account for the periodicity of the domain, but the effect should not be noticeable
        in any generated outputs.

    Args:
        val: The (velocty) vector field.
        scales: Optional :py:class:`Scales` object.
                If provided, the array will be rescaled from lattice to physical units.

    Returns:
        The curl of the vector field.
    """

    # Check the shape of the input array
    if val.ndim == 3:
        assert val.shape[-1] == 2, f"Inconsistent shape! Expected a 2D vector field. {val.shape}"
        ndim = 2
    elif val.ndim == 4:
        assert val.shape[-1] == 3, f"Inconsistent shape! Expected a 3D vector field. {val.shape}"
        ndim = 3
    else:
        raise ValueError(f"Invalid array dimensionality! {val.shape}")

    grad = np.gradient(val)

    if ndim == 2:
        out = grad[1][..., 0] - grad[0][..., 1]
    else:
        out = np.stack(
            (
                grad[2][..., 1] - grad[1][..., 2],
                grad[0][..., 2] - grad[2][..., 0],
                grad[1][..., 0] - grad[0][..., 1],
            ),
            axis=-1,
        )

    if scales is not None:
        out[:] = scales.converter(T=-1).to_physical_units(out)

    return out


def calc_stream_func(
    vel: np.ndarray,
    scales: Scales | None = None,
    origin: tuple[int, int] | tuple[float, float] | None = None,
) -> np.ndarray:
    """
    Calculate the 2D `stream function <https://en.wikipedia.org/wiki/Stream_function>`_ of a velocity field.

    Note:
        The reference point is at the (0, 0) index of the array.

    Args:
        val: The (velocty) vector field.
        scales: Optional :py:class:`Scales` object.
                If provided, the array will be rescaled from lattice to physical units.
        origin:

    Returns:
        The stream function.
    """

    assert vel.ndim == 3 and vel.shape[2] == 2, "Expected 2D velocity field!"

    origin = origin or (0, 0)

    if scales is not None:
        origin = (
            scales.distance.to_lattice_units(origin[0]),
            scales.distance.to_lattice_units(origin[1]),
        )

    x, y = origin
    x = int(x)
    y = int(y)

    ivdx = np.cumsum(vel[..., 1], axis=0)
    iudy = np.cumsum(vel[..., 0], axis=1)

    phi = iudy[0, :][None, :] - ivdx
    phi -= phi[x, y]

    if scales is not None:
        phi[:] = scales.converter(L=2, T=-1).to_physical_units(phi)

    return phi
