import vtk
import ngsolve
import time

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk


def load_volume_mesh(path):
    print(f'file path : {path}')
    return ngsolve.Mesh(path)


def addScalarCellData(triangle_polydata, cell_data, components, name):
    t_start = time.time()
    points_array = numpy_to_vtk(cell_data, deep=True)
    points_array.SetName(name)
    triangle_polydata.GetPointData().AddArray(points_array)
    print("copy time : ", time.time() - t_start)

    return triangle_polydata


def iglToVtkPolydata(sf, sv):
    triangle_polydata = vtk.vtkPolyData()

    points_array = numpy_to_vtk(sv, deep=True)

    _sf = np.array(sf)
    nbpts = np.full(_sf.shape[0], 3)
    _sf = np.column_stack((nbpts, _sf))

    triangles_array = numpy_to_vtk(_sf, deep=True, array_type=vtk.VTK_ID_TYPE)

    points = vtk.vtkPoints()
    points.SetData(points_array)

    cells2 = vtk.vtkCellArray()
    cells2.SetCells(triangles_array.GetNumberOfTuples(), triangles_array)

    # Fügen Sie die Punkte und Zellen zur PolyData hinzu
    triangle_polydata.SetPoints(points)
    triangle_polydata.SetPolys(cells2)
    return triangle_polydata


def ngsolve_result_to_vtkpolydata(mesh, gfu, f):
    eigenmodes = [0]*len(gfu.vecs)

    time_start = time.time()
    vertices = [ [p[0], p[1], p[2]] for p in mesh.ngmesh.Points() ]
    print("point copy : ", time.time() - time_start)

    time_start = time.time()
    triangles2 = [(t[0][0:3] - 1).tolist() for t in np.array(mesh.ngmesh.Elements2D())]
    print("triangle copy : ", time.time() - time_start)

    time_start = time.time()
    polyData = iglToVtkPolydata(triangles2, vertices)
    print("polydata copy : ", time.time() - time_start)

    time_start = time.time()
    meshpoints = [mesh(v[0], v[1], v[2]) for v in vertices]
    print("meshpoints generated : ", time.time() - time_start)

    for k in range(len(f)):
        E = gfu.MDComponent(k)
        name = str(f[k]) + "Hz"
        time_start = time.time()
        eigenmodes[k] = [ E.real(x) for x in meshpoints ]
        print(name + " extract : ", time.time() - time_start)
        polyData = addScalarCellData(polyData, eigenmodes[k], 3, name)

    return polyData


def save_ngsolve_result_as_vtk(filepath, mesh, u, f):

    polyData = ngsolve_result_to_vtkpolydata(mesh, u, f)

    appendFilter = vtk.vtkAppendFilter()
    appendFilter.AddInputData(polyData)
    appendFilter.Update()

    unstructuredGrid = vtk.vtkUnstructuredGrid()
    unstructuredGrid.ShallowCopy(appendFilter.GetOutput())

    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileVersion(vtk.vtkUnstructuredGridWriter.VTK_LEGACY_READER_VERSION_4_2)
    writer.SetFileName(filepath)
    writer.SetInputData(unstructuredGrid)
    writer.Write()
