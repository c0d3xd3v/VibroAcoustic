import os
import sys
import igl
import numpy as np
from pyFloatTetwildWrapper import FTetWildWrapper

from ngsolve_tools import *


def scale_to_box(V, target_size=1000.0):
    bb_min = V.min(axis=0)
    bb_max = V.max(axis=0)
    extent = bb_max - bb_min
    max_extent = extent.max()

    scale = target_size / max_extent
    V_scaled = (V - bb_min) * scale

    return V_scaled, scale, bb_min

def unscale(V_scaled, scale, translation):
    return V_scaled / scale + translation

if __name__ == "__main__":
    path = sys.argv[1]

    file_path = os.path.dirname(os.path.abspath(path))
    file_name, file_ext = os.path.splitext(os.path.basename(path))

    V, F = igl.read_triangle_mesh(path)

    tw = FTetWildWrapper(stop_energy=10, ideal_edge_length_rel=0.075, eps_rel=0.0005)
    tw.loadMeshGeometry(V, F)
    tw.tetrahedralize()

    tris, tets, nods = tw.getSurfaceIndices()

    #tw.save(file_path + "/" + file_name + ".msh")
    saveNgSolveMesh(nods, tris, tets, file_path + "/" + file_name + ".vol")
