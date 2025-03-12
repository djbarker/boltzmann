import numpy as np

from boltzmann.units import Scales


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
        out /= scales.dt

    return out
