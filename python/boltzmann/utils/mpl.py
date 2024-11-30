import numpy as np
import matplotlib.pyplot as plt

from typing import Any

from matplotlib.figure import Figure

from boltzmann.impl2 import PeriodicDomain, unflatten
from boltzmann.core import DomainMeta


class PngWriter(object):

    def __init__(
        self,
        path: str,
        domain: DomainMeta,
        pidx: PeriodicDomain,
        cell: np.ndarray,
        data: np.ndarray,
        cmap: str,
        vmin: float,
        vmax: float,
        fig_kwargs: dict[str, Any],
    ):

        cell = unflatten(pidx, cell)[1:-1, 1:-1]
        data = unflatten(pidx, data)[1:-1, 1:-1]
        data = np.where(cell == 1, np.nan, data)

        self.data = data
        self.pidx = pidx

        ey = domain.extent_si[0, 1] - domain.extent_si[0, 0]
        ex = domain.extent_si[1, 1] - domain.extent_si[1, 0]
        ar = ey / ex

        fig = plt.figure(figsize=(8, 8 / ar), **fig_kwargs)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, interpolation="none")

        plt.axis("off")

        self.path = path
        self.fig = fig

    def __enter__(self) -> Figure:
        return self.fig

    def __exit__(self, *a, **k):
        self.fig.savefig(self.path, dpi=200, bbox_inches="tight", pad_inches=0)
        self.fig.clear()
        plt.close()
