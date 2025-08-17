import os
import vtk
import ngsolve
import time

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk


class Pathinformations():
    pfad: str
    dateiname_mit_endung: str
    dateiname: str
    endung: str
    absoluter_pfad: str
    verzeichnis: str

    def __init__(self, filepath):
        self.pfad = os.path.dirname(filepath)
        self.dateiname_mit_endung = os.path.basename(filepath)
        self.dateiname, self.endung = os.path.splitext(self.dateiname_mit_endung)
        self.absoluter_pfad = os.path.abspath(filepath)
        self.verzeichnis = os.path.dirname(self.absoluter_pfad)

    def formated_info(self):
        info = (
            f"Pfad: {self.pfad}\n"
            f"Dateiname: {self.dateiname}\n"
            f"Endung: {self.endung}\n"
            f"Verzeichnis: {self.verzeichnis}"
        )
        return info

    def __str__(self):
        return self.formated_info()


def split_path_informations(filepath):
    return Pathinformations(filepath)


def load_volume_mesh(path):
    print(f'file path : {path}')
    ngs_mesh = ngsolve.Mesh(path)
    
    return ngs_mesh


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

    triangle_polydata.SetPoints(points)
    triangle_polydata.SetPolys(cells2)
    return triangle_polydata


def ngsolve_result_to_vtkpolydata(mesh, gfu, f):
    eigenmodes = [0]*len(gfu.vecs)

    time_start = time.time()
    vertices = [ [p[0], p[1], p[2]] for p in mesh.ngmesh.Points() ]

    time_start = time.time()
    triangles2 = [(t[0][0:3] - 1).tolist() for t in np.array(mesh.ngmesh.Elements2D())]

    time_start = time.time()
    polyData = iglToVtkPolydata(triangles2, vertices)

    time_start = time.time()
    meshpoints = [mesh(v[0], v[1], v[2]) for v in vertices]

    for k in range(len(gfu.vecs)):
        E = gfu.MDComponent(k)
        #name = "eigenmode" + str(k)
        name = str(round(f[k].real, 2))+"Hz"
        time_start = time.time()
        #eigenmodes[k] = [ E.real(x) for x in meshpoints ]

        # Nur Realanteil extrahieren
        mode_values = np.array([E.real(x) for x in meshpoints])
        
        # Norm jedes Vektors berechnen
        norms = np.linalg.norm(mode_values, axis=1)
        max_norm = np.max(norms)
        
        # Normalisieren auf max = 1
        if max_norm != 0:
            mode_values /= max_norm
        
        eigenmodes[k] = mode_values

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
