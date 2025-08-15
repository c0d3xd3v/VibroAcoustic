import os

import ngsolve
from ngsolve import *
from ngsolve.fem import NODE_TYPE

from fileio import load_volume_mesh
from fileio import save_ngsolve_result_as_vtk

import numpy as np



print(f'ngsolve {ngsolve.__version__}')
# Geometriedaten
filepath = sys.argv[1]

pfad = os.path.dirname(filepath)
dateiname_mit_endung = os.path.basename(filepath)
dateiname, endung = os.path.splitext(dateiname_mit_endung)
absoluter_pfad = os.path.abspath(filepath)
verzeichnis = os.path.dirname(absoluter_pfad)

print("Pfad:", pfad)
print("Dateiname:", dateiname)
print("Endung:", endung)
print("Verzeichnis:", verzeichnis)

mesh = load_volume_mesh(filepath)

rho = 2.535e-6  # kg/mm³
E = 6.90e5      # N/mm²
nu = 0.33       # dimensionslos
mu_val = E / (2 * (1 + nu))
lam = E * nu / ((1 + nu) * (1 - 2*nu))

print(f'rho : {rho}')
print(f'E   : {E}')
print(f'nu  : {nu}')

vertices = mesh.ngmesh.Points()

# Separiere x, y, z Koordinaten
x_coords = [p[0] for p in vertices]
y_coords = [p[1] for p in vertices]
z_coords = [p[2] for p in vertices]

xmin, ymin, zmin = min(x_coords), min(y_coords), min(z_coords)
xmax, ymax, zmax = max(x_coords), max(y_coords), max(z_coords)

V = Integrate(1, mesh)
V_box = (xmax - xmin)*(ymax - ymin)*(zmax - zmin)
L = V**(1/3)
c = np.sqrt(E/rho)

print(f"dx: ({xmax - xmin})")
print(f"dy: ({ymax - ymin})")
print(f"dz: ({zmax - zmin})")
print(f'V : {V}')
print(f'V_box : {V_box}')
print(f'1.0 - (V_box - V)/V_box : {1.0 - (V_box - V)/V_box:.2e}')
print("Masse des Körpers:", rho*V, "kg")
print(f'characteristic length L_c : {L}')
print(f'c : {c}')
print(f'c/L : {c/(2*L)} ')
