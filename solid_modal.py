import sys

from fileio import load_volume_mesh
from fileio import save_ngsolve_result_as_vtk

from material_definitons import steel, aluminium

from eigenfrequencies import build_simple_solid_fes
from eigenfrequencies import solve_elasticity_system


filepath = sys.argv[1]
mesh = load_volume_mesh(filepath)

solid_fes = build_simple_solid_fes(mesh)
u, f = solve_elasticity_system(steel, solid_fes)

save_ngsolve_result_as_vtk("output.vtk", mesh, u)
