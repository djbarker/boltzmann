from __future__ import annotations

import numpy as np

from math import sqrt


__all__ = [
    "CellFlags",
    "calc_lbm_params",
    "check_lbm_params",
]

# ------------------------------------
# Re-export the rust module here.

import boltzmann.boltzmann  # type: ignore
from boltzmann.boltzmann import *  # type: ignore  # noqa: F403

if hasattr(boltzmann.boltzmann, "__all__"):
    __all__.extend(boltzmann.boltzmann.__all__)

# ------------------------------------


class CellFlags:
    FLUID = 0
    WALL = 1
    FIXED_FLUID_VELOCITY = 2
    FIXED_FLUID_DENSITY = 4
    FIXED_SCALAR_VALUE = 8
    FIXED_FLUID = FIXED_FLUID_VELOCITY | FIXED_FLUID_DENSITY


def check_lbm_params(Re: float, L: float, tau: float, M_max: float = 0.1):
    """
    Check if the chosen parameters are likely to be stable or not with BGK collision operator.
    """

    # Maximum value of tau implied by Mach condition.
    tau_mach = np.sqrt(3) * M_max * L / Re + 0.5

    # Minimum value of L implied by the BGK stability condition.
    L_stab = Re / 24

    eps = 1e-8
    assert L + eps > L_stab, f"BGK stability condition violated. {L=} < {L_stab}"
    assert tau < tau_mach + eps, f"Mach condition violated. {tau=} > {tau_mach}"


def calc_lbm_params(
    # params
    Re: float,
    L: float,
    tau: float | None = None,
    # bounds
    M_max: float = 0.1,
    tau_max: float = 1.0,
    tau_min: float = 0.5,
) -> tuple[float, float]:
    """
    Choose the values of tau and u which maximize simulation speed & verify stability.
    If you specify tau that value will be used instead.
    """

    if tau is None:
        tau = sqrt(3) * M_max * L / Re + 0.5  # Maximum permissible by Mach condition
        tau = min(max(tau, tau_min), tau_max)  # Hard bounds on tau

    check_lbm_params(Re, L, tau, M_max)

    nu = (tau - 0.5) / 3
    u = Re * nu / L

    return (tau, u)
