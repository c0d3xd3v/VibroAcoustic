import sys

from ngsolve import SymbolicBFI
from ngsolve import VectorH1
from ngsolve import BilinearForm
from ngsolve import InnerProduct
from ngsolve import Preconditioner
from ngsolve import GridFunction
from ngsolve import ArnoldiSolver
from ngsolve import div, grad, Id
from ngsolve.krylovspace import EigenValues_Preconditioner

from fileio import load_volume_mesh
from fileio import save_ngsolve_result_as_vtk

from material_definitons import steel, aluminium, Material
from Material import aluminium as mm

import numpy as np

filepath = sys.argv[1]
mesh = load_volume_mesh(filepath)
rho = 2.5355e-6   # kg/mm³
E   =  6.9e7     # N/mm²
nu = 0.33 
_E = mm.E("mm")
_rho = mm.rho("mm")

material = Material("linear", E, nu, rho)
count = 12

# Tensoren
def eps(w): return 0.5*(grad(w)+grad(w).trans)
def sigma(w, material): return material.lam*div(w)*Id(3) + 2*material.mu*eps(w)

solid_fes = VectorH1(mesh, order=2, dirichlet=[], complex=True)
u, v = solid_fes.TrialFunction(), solid_fes.TestFunction()

# ngsolve.BilinearForm(fes, symmetric=True, eliminate_internal=True)
a = BilinearForm(solid_fes, symmetric=True)
a += SymbolicBFI(InnerProduct(sigma(u, material), eps(v)))

b = BilinearForm(solid_fes, symmetric=True)
b += SymbolicBFI(material.rho * InnerProduct(u, v))

pre = Preconditioner(a, type="h1amg")  # oder "direct", etc.

a.Assemble()
b.Assemble()

pre.Update()

lams = EigenValues_Preconditioner(mat=a.mat, pre=pre)
lam_list = [abs(float(l)) for l in lams]
filtered_lams = [l for l in lam_list if l > 1e-12]
if not filtered_lams:
    kappa = float('inf')
else:
    kappa = max(filtered_lams) / min(filtered_lams)

print(f"Konditionszahl: {kappa:.2e}")

u = GridFunction(solid_fes, multidim=count)
lams = ArnoldiSolver(a.mat, b.mat, solid_fes.FreeDofs(), list(u.vecs), 1000, inverse="pardiso")
f = np.sqrt(lams)/(2.0 * np.pi)

save_ngsolve_result_as_vtk("output.vtk", mesh, u, f)
