import numpy as np


def vortex(
    xx: np.ndarray,
    yy: np.ndarray,
    x0: float,
    y0: float,
    vx: float,
    vy: float,
    vmax: float,
    sigma: float,
    clockwise: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Util function to provide an initial velocity field which looks like a vortex.
    """

    dxp = xx - x0
    dyp = yy - y0
    r = np.sqrt(dxp * dxp + dyp * dyp) / sigma
    # v = (2 * r / (sigma * sigma)) * np.exp(-r * r / (sigma * sigma))
    v = r * np.exp(1 - r)
    theta = np.arctan2(dyp, dxp)

    sign = 1 if clockwise else -1
    vxout = +vmax * v * np.sin(theta) * sign + np.exp(-r * r) * vx
    vyout = -vmax * v * np.cos(theta) * sign + np.exp(-r * r) * vy

    return vxout, vyout
