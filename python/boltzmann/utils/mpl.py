import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from typing import Any

from boltzmann.core import Domain


class PngWriter(object):
    """
    Save 2D scalar fields to a PNG image.

    Matplotlib is very slow therefore this uses Pillow under the hood.
    We can have some very basic annotation.
    """

    def __init__(
        self,
        path: str | Path,
        domain: Domain,
        cell: np.ndarray,
        data: np.ndarray,
        cmap: str,
        vmin: float,
        vmax: float,
    ):
        self.path = path

        cell = domain.unflatten(cell).T[::-1, :]
        data = domain.unflatten(data).T[::-1, :]
        data = np.where(cell == 1, np.nan, data)
        data = (data - vmin) / (vmax - vmin)
        data = data.clip(vmin, vmax)

        cmap_ = plt.get_cmap(cmap)
        self.img = Image.fromarray((cmap_(data)[:, :, :3] * 255).astype(np.uint8))

    def __enter__(self) -> Image.Image:
        return self.img

    def __exit__(self, *a, **k):
        self.img.save(self.path)
