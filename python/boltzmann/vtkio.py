import logging
from pathlib import Path
from typing import Iterable
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from boltzmann.core import Domain


logger = logging.getLogger(__name__)


def to_vtk_type(np_dtype: np.dtype) -> int:
    """
    Convert the Numpy dtype to the correct VTK flag.
    """
    match np_dtype:
        case np.int32 | np.int64:
            return vtk.VTK_INT
        case np.float32 | np.float64:
            return vtk.VTK_FLOAT
        case _:
            raise ValueError(f"Unknown VTK type for Numpy dtype {np_dtype!r}")


class VtiWriter:
    """
    Class for outputting the simulation data to a VTK .vti file.
    Can be used as a context manager.

    .. code-block:: python

        with VtiWriter("output.vti", dom) as writer:
            writer.add_data("velocity", vel)
            writer.add_data("density", rho, default=True)
    """

    def __init__(self, path: str | Path, counts: Iterable[int]):
        """
        :param path: The path to write the VTI file to.
        :param counts: The grid size in each dimension.
        """
        self.path = str(path)
        self.data: dict[str, np.ndarray] = {}  # The field data. A name -> data mapping.
        self.counts = list(counts)
        self.default: str | None = None  # Name of the field displayed by default in ParaView.

    @property
    def ndim(self) -> int:
        return len(self.counts)

    def add_data(self, name: str, val: np.ndarray):
        """
        :param name: The name of the field as it will appear in ParaView.
        :param val: The field data.
        """
        assert list(val.shape[: self.ndim]) == self.counts

        def _adjust_shape(v: np.ndarray):
            """
            VTK needs vectors to be 3D, and scalars to be "un-squashed" (in the Numpy sense).
            Adds an extra axis to scalar arrays, and pads 1D & 2D vectors with zeros.
            Leaves 3D vector, and tensor, arrays untouched.
            """
            # scalar array
            if list(v.shape) == self.counts:
                return v[..., None]

            # vector arrays
            if len(v.shape) == self.ndim + 1:
                match v.shape[-1]:
                    case 1:
                        return np.pad(v, [(0, 0)] * self.ndim + [(0, 2)])
                    case 2:
                        return np.pad(v, [(0, 0)] * self.ndim + [(0, 1)])
                    case 3:
                        return v
                    case _:
                        raise ValueError(f"Invalid dimension! {v.shape[-1]}")

            # tensor array
            return v

        self.data[name] = _adjust_shape(val)

    def set_default(self, name: str):
        """
        Sets the field which is displayed by default in ParaView.
        If not set the first field added will be the default.
        If it has already been set to a different field this will raise a `ValueError`.

        :param name: The field name to set as default.
        """

        if self.default is not None and self.default != name:
            raise ValueError(f"Default key already set! [old={self.default!r}, new={name!r}]")

        self.default = name

    def write(self):
        """
        Save the VTI file.
        """

        def _to3d_counts(c: list[int]) -> list[int]:
            if len(c) > 3:
                raise ValueError(f"Invalid dimension! {len(c)!r}")
            return c + [1] * (3 - len(c))

        nx, ny, nz = _to3d_counts(self.counts)

        img_data = vtk.vtkImageData()
        img_data.SetDimensions(nx, ny, nz)
        pnt_data = img_data.GetPointData()

        for name, np_data in self.data.items():
            vtk_type = to_vtk_type(np_data.dtype)
            vtk_data = numpy_to_vtk(
                num_array=np.swapaxes(np_data, 0, self.ndim - 1).ravel(order="C"),
                deep=False,
                array_type=vtk_type,
            )
            vtk_data.SetName(name)
            vtk_data.SetNumberOfComponents(np_data.shape[-1])
            pnt_data.AddArray(vtk_data)

        if self.default is not None:
            pnt_data.SetActiveAttribute(self.default, vtk.VTK_ATTRIBUTE_MODE_DEFAULT)

        # zipper = vtk.vtkZLibDataCompressor()
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(self.path)
        # writer.SetCompressor(zipper)
        writer.SetInputData(img_data)
        writer.Write()

    def __enter__(self) -> "VtiWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.warning(f"Not writing VTK file due to exception. [path={self.path!r}]")
            return

        self.write()
