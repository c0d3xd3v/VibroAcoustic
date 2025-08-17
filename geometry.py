from numpy import min
from numpy import max

from ngsolve import Integrate


def compute_characteristic_lenght(mesh):    
    V = Integrate(1, mesh)
    L = V**(1/3)
    return L

def compute_bounding_box_volume(mesh):
    vertices = mesh.ngmesh.Points()

    x_coords = [p[0] for p in vertices]
    y_coords = [p[1] for p in vertices]
    z_coords = [p[2] for p in vertices]

    xmin, ymin, zmin = min(x_coords), min(y_coords), min(z_coords)
    xmax, ymax, zmax = max(x_coords), max(y_coords), max(z_coords)

    return (xmax - xmin)*(ymax - ymin)*(zmax - zmin)
