import os

import ngsolve
from ngsolve import *
from ngsolve.fem import NODE_TYPE

from fileio import load_volume_mesh
from fileio import save_ngsolve_result_as_vtk

from geometry import compute_characteristic_lenght
from geometry import compute_bounding_box_volume

import numpy as np



print(f'ngsolve {ngsolve.__version__}')
# Geometriedaten
filepath = sys.argv[1]
mesh = load_volume_mesh(filepath)

rho = 2.535e-6  # kg/mm³
E = 6.90e5      # N/mm²
nu = 0.33       # dimensionslos
mu_val = E / (2 * (1 + nu))
lam = E * nu / ((1 + nu) * (1 - 2*nu))

#print(f'rho : {rho}')
#print(f'E   : {E}')
#print(f'nu  : {nu}')

L = compute_characteristic_lenght(mesh)
V_box = compute_bounding_box_volume(mesh)
V = L**3

c = np.sqrt(E/rho)

print(f'1.0 - (V_box - V)/V_box : {1.0 - (V_box - V)/V_box:.2e}')
print("Masse des Körpers:", rho*V, "kg")
print(f'characteristic length L_c : {L}')
print(f'c : {c}')
print(f'c/L : {c/(2*L)} ')
