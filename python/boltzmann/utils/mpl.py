from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from typing import Any

from matplotlib.figure import Figure

from boltzmann.core import Domain


class PngWriter(object):
    def __init__(
        self,
        path: str | Path,
        domain: Domain,
        cell: np.ndarray,
        data: np.ndarray,
        cmap: str,
        vmin: float,
        vmax: float,
        fig_kwargs: dict[str, Any],
    ):
        cell = domain.unflatten(cell)[1:-1, 1:-1]
        data = domain.unflatten(data)[1:-1, 1:-1]
        data = np.where(cell == 1, np.nan, data)

        self.data = data

        ey = domain.upper[0] - domain.lower[0]
        ex = domain.upper[1] - domain.lower[1]
        ar = ey / ex

        fig = plt.figure(figsize=(10, 10 / ar), **fig_kwargs)
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
