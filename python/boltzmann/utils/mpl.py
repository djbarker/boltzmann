import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import Colormap
from PIL.Image import Image, Resampling, fromarray
from pathlib import Path
from typing import Any

from boltzmann.core import Domain


class PngWriter(object):
    """
    Save 2D scalar fields, or RGB color fields to a PNG image.

    Matplotlib is very slow therefore this uses Pillow under the hood.
    We can have some very basic annotation.
    """

    def __init__(
        self,
        path: str | Path,
        outx: int,
        domain: Domain,
        cell: np.ndarray,
        data: np.ndarray,
        cmap: str | Colormap | None = None,
        vmin: float = 0.0,
        vmax: float = 1.0,
    ):
        self.path = path

        data = domain.unflatten(data)
        cell = domain.unflatten(cell)
        cell = cell[:, :, None]
        if data.ndim == 2:
            data = data[:, :, None]

        data = np.where(cell == 1, np.nan, data)
        data = (data - vmin) / (vmax - vmin)
        data = np.clip(data, 0.0, 1.0)
        data = np.swapaxes(data, 0, 1)

        if data.shape[-1] != 3:
            assert cmap is not None
            cmap_ = plt.get_cmap(cmap)
            cmap_.set_bad("#CCCCCC")
            data = np.squeeze(cmap_(data)[::-1, :, 0, :3])
        else:
            data = data[::-1, :, :]
            data[~np.isfinite(data)] = 0xDC / 255

        self.img = fromarray((data * 255).astype(np.uint8))

        ar = domain.counts[1] / domain.counts[0]
        outy = int(outx * ar)
        self.outxy = (outx, outy)

    def __enter__(self) -> Image:
        return self.img

    def __exit__(self, *a, **k):
        self.img.thumbnail(self.outxy, Resampling.LANCZOS)
        self.img.save(self.path)
