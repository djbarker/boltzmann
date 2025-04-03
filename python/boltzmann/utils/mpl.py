import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import LinearSegmentedColormap, Colormap
from PIL.Image import Image, Resampling, fromarray
from pathlib import Path

__all__ = [
    "make_cmap",
    "OrangeBlue",
    "OrangeBlue_r",
    "InkyBlueRed",
    "InkyBlueRed_r",
    "PngWriter",
]


def make_cmap(name: str, colors: list) -> LinearSegmentedColormap:
    """
    Make a custom colormap from a list of colors which are evenly spaced.
    """
    nodes = np.linspace(0, 1, len(colors))
    return LinearSegmentedColormap.from_list(name, list(zip(nodes, colors)))


_colors = [
    "#ffe359",
    "#ff8000",
    "#734c26",
    "#17151a",
    "#17151a",
    "#17151a",
    "#265773",
    "#0f82b8",
    "#8fceff",
]

_nodes = [0.0, 0.16666667, 0.33333333, 0.49, 0.5, 0.51, 0.66666667, 0.83333333, 1.0]

OrangeBlue = LinearSegmentedColormap.from_list("OrangeBlue", list(zip(_nodes, _colors)))
OrangeBlue_r = LinearSegmentedColormap.from_list("OrangeBlue_r", list(zip(_nodes, _colors[::-1])))

_colors = [
    "#10396a",
    "#1968c2",
    "#198fe3",
    "#19cdff",
    "#ffffff",
    "#ffffff",
    "#ffffff",
    "#ffaf18",
    "#f3681a",
    "#e32b2b",
    "#890000",
]

_nodes = [0.0, 0.125, 0.25, 0.375, 0.49, 0.5, 0.51, 0.625, 0.75, 0.875, 1.0]

InkyBlueRed = LinearSegmentedColormap.from_list("InkyBlueRed", list(zip(_nodes, _colors)))
InkyBlueRed_r = LinearSegmentedColormap.from_list("InkyBlueRed_r", list(zip(_nodes, _colors[::-1])))

# Register our custom colourmaps.
mpl.colormaps.register(cmap=OrangeBlue)
mpl.colormaps.register(cmap=OrangeBlue_r)
mpl.colormaps.register(cmap=InkyBlueRed)
mpl.colormaps.register(cmap=InkyBlueRed_r)
mpl.colormaps.register(cmap=InkyBlueRed, name="InkyRedBlue_r")
mpl.colormaps.register(cmap=InkyBlueRed_r, name="InkyRedBlue")


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
        cell: np.ndarray,
        data: np.ndarray,
        cmap: str | Colormap | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
    ):
        vmin = vmin or np.nanmin(data)
        vmax = vmax or np.nanmax(data)

        self.path = path

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
            data[~np.isfinite(data)] = 0xAC / 255

        self.img = fromarray((data * 255).astype(np.uint8))

        # NOTE: We have already tranposed here, so y is in position 0.
        ar = data.shape[0] / data.shape[1]
        outy = int(outx * ar)
        self.outxy = (outx, outy)

    def __enter__(self) -> Image:
        return self.img

    def __exit__(self, *a, **k):
        if self.outxy[0] != self.img.width or self.outxy[1] != self.img.height:
            self.img.thumbnail(self.outxy, Resampling.LANCZOS)
        self.img.save(self.path)
