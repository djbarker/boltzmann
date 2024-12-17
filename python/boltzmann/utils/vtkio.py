import numpy as np

import vtk
import vtk.util.numpy_support as vtk_np

from boltzmann.core import Domain, CellType


def write_vti(
    path: str,
    dom: Domain,
    v: np.ndarray,
    rho: np.ndarray,
    curl: np.ndarray,
    cell: np.ndarray,
):
    # need 3d vectors for vtk
    def _to3d(v: np.ndarray):
        if v.shape[-1] == 2:
            return np.pad(v, [(0, 0), (0, 0), (0, 1)])
        else:
            return v

    # transpose data for writing
    v_T = np.copy(np.transpose(_to3d(v), (1, 0, 2)))
    rho_T = np.copy(rho.T)
    curl_T = np.copy(curl.T)
    cell_T = np.copy(cell.T)

    v_T[cell_T == CellType.BC_WALL.value, :] = np.nan
    rho_T[cell_T == CellType.BC_WALL.value] = np.nan
    curl_T[cell_T == CellType.BC_WALL.value] = np.nan

    # cut off periodic part
    v_T = v_T[1:-1, 1:-1]
    rho_T = rho_T[1:-1, 1:-1]
    curl_T = curl_T[1:-1, 1:-1]
    cell_T = cell_T[1:-1, 1:-1]

    image_data = vtk.vtkImageData()
    nx, ny = list(dom.counts)
    image_data.SetDimensions(nx, ny, 1)

    rho_data = vtk_np.numpy_to_vtk(num_array=rho_T.ravel(), deep=False, array_type=vtk.VTK_FLOAT)
    rho_data.SetName("Density")
    rho_data.SetNumberOfComponents(1)

    vel_data = vtk_np.numpy_to_vtk(num_array=v_T.ravel(), deep=False, array_type=vtk.VTK_FLOAT)
    vel_data.SetName("Velocity")
    vel_data.SetNumberOfComponents(3)

    curl_data = vtk_np.numpy_to_vtk(num_array=curl_T.ravel(), deep=False, array_type=vtk.VTK_FLOAT)
    curl_data.SetName("Vorticity")
    curl_data.SetNumberOfComponents(1)

    wall_data = vtk_np.numpy_to_vtk(num_array=cell_T.ravel(), deep=False, array_type=vtk.VTK_FLOAT)
    wall_data.SetName("CellType")
    wall_data.SetNumberOfComponents(1)

    p_data = image_data.GetPointData()
    p_data.AddArray(rho_data)
    p_data.AddArray(vel_data)
    p_data.AddArray(curl_data)
    p_data.AddArray(wall_data)

    p_data.SetActiveAttribute("Velocity", vtk.VTK_ATTRIBUTE_MODE_DEFAULT)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(path)
    writer.SetInputData(image_data)
    writer.Write()
