import logging
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from boltzmann.impl2 import PeriodicDomain, NumbaParams, unflatten


log = logging.getLogger(__name__)


# need 3d vectors for vtk
def _to3d(v: np.ndarray):
    if v.shape[-1] == 2:
        return np.pad(v, [(0, 0), (0, 1)])
    else:
        return v


def to_vtk_type(np_dtype: np.dtype) -> int:
    match np_dtype:
        case np.int32 | np.int64:
            return vtk.VTK_INT
        case np.float32 | np.float64:
            return vtk.VTK_FLOAT
        case _:
            raise ValueError(f"Unknown VTK type for Numpy dtype {np_dtype!r}")


class VtiWriter:

    def __init__(self, path: str, pidx: PeriodicDomain):
        self.path = path
        self.pidx = pidx
        self.data = {}
        self.default = None

    def add_data(self, key: str, val: np.ndarray, *, default: bool = False):
        assert val.shape[0] == np.prod(self.pidx.counts)
        assert np.ndim(val) <= 2
        if np.ndim(val) == 1:
            val = val[:, None]
        self.data[key] = unflatten(self.pidx, _to3d(val), rev=False)

        match self.default, default:
            case None, True:
                self.default = key
            case str(), True:
                raise ValueError(f"Default key already set! [old={self.default!r}, new={key!r}]")

    def write(self):
        counts = self.pidx.counts

        img_data = vtk.vtkImageData()
        nx, ny = list(counts)
        img_data.SetDimensions(nx, ny, 1)

        pnt_data = img_data.GetPointData()

        for name, np_data in self.data.items():
            vtk_type = to_vtk_type(np_data.dtype)
            vtk_data = numpy_to_vtk(num_array=np_data.ravel(), deep=False, array_type=vtk_type)
            vtk_data.SetName(name)
            vtk_data.SetNumberOfComponents(np_data.shape[-1])
            pnt_data.AddArray(vtk_data)

        if self.default is not None:
            pnt_data.SetActiveAttribute(self.default, vtk.VTK_ATTRIBUTE_MODE_DEFAULT)

        zipper = vtk.vtkZLibDataCompressor()
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(self.path)
        writer.SetCompressor(zipper)
        writer.SetInputData(img_data)
        writer.Write()

    def __enter__(self) -> "VtiWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            log.warning(f"Not writing VTK file due to exception. [path={self.path!r}]")
            return

        self.write()
