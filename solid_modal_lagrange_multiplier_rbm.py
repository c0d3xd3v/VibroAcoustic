import os

import ngsolve
from ngsolve import *

from fileio import load_volume_mesh
from fileio import save_ngsolve_result_as_vtk
from fileio import split_path_informations

from geometry import compute_bounding_box

from Material import Material

from fem_utils import unit_check
from fem_utils import rigid_body_modes
from fem_utils import compute_condition_number

from eigen_solvers import solve_scalar_eigen_problem_a
from eigen_solvers import solve_scalar_eigen_problem_p

import numpy as np


if __name__ == "__main__":
    print(f'ngsolve {ngsolve.__version__}')
    # Geometriedaten
    filepath = sys.argv[1]

    path_info = split_path_informations(filepath)
    print(path_info)

    mesh = load_volume_mesh(filepath)

    # FE-Räume
    V = VectorH1(mesh, order=2, complex=True)  # Verschiebung (3D)
    Q = [NumberSpace(mesh, complex=True) for _ in range(6)]  # Lagrange-Multiplikatoren für 6 RBMs
    X = FESpace([V] + Q)  # Gemischter Raum

    # Trial- und Testfunktionen
    (u, *lambdas), (v, *mus) = X.TrialFunction(), X.TestFunction()

    material = Material("linear", 6.9e7, 0.33, 2.5355e-6, units="mm")
    print(material)
    '''
    rho = 2.5355e-6   # kg/mm³
    E   =  6.9e7     # N/mm²

    unit_check(rho, E, "mm")

    nu = 0.33       # dimensionslos
    mu_val = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2*nu))

    print(f'rho : {rho}')
    print(f'E   : {E}')
    print(f'nu  : {nu}')
    '''
    compute_bounding_box(mesh)

    # Tensors
    def eps(w): return 0.5*(grad(w)+grad(w).trans)
    def sigma(w): return material.lam*div(w)*Id(3) + 2*material.mu*eps(w)

    rbm_gfs_ortho = rigid_body_modes(V, material.rho())

    # Bilinearform
    a = BilinearForm(X, symmetric=True)
    a += SymbolicBFI(InnerProduct(sigma(u), eps(v)))

    rbm_coeffs = [CoefficientFunction(rbm_gf) for rbm_gf in rbm_gfs_ortho]
    for i in range(len(rbm_gfs_ortho)):
        a += SymbolicBFI(InnerProduct(rbm_coeffs[i], v) * lambdas[i])
        a += SymbolicBFI(InnerProduct(rbm_coeffs[i], u) * mus[i])

    b = BilinearForm(X, symmetric=True)
    b.components[0] += SymbolicBFI(material.rho()*InnerProduct(V.TrialFunction(), V.TestFunction()))
    for i in range(1, len(rbm_gfs_ortho)+1):
        b.components[i] += SymbolicBFI(material.rho()*CoefficientFunction(0.0))

    pre = ngsolve.Preconditioner(a, type="direct")  # oder "bddc, h1amg, direct", etc.

    a.Assemble()
    b.Assemble()

    pre.Update()

    print(f"Konditionszahl(pre x mat) : {compute_condition_number(a, pre):.2e}")

    # Lösung
    lams, u = solve_scalar_eigen_problem_a(material.rho(), X, a.mat, b.mat, rbm_gfs_ortho, 12)
    f = np.sqrt(lams) / (2.0 * np.pi)

    for i, rbm in enumerate(rbm_gfs_ortho):
        val = Integrate(material.rho() * InnerProduct(rbm, u.components[0]), mesh)
        print(f"Lagrange-Kopplung {i}: {val}")

    u_vec = u.components[0].vec.Copy()

    for rbm in rbm_gfs_ortho:
        coeff = Integrate(material.rho() * InnerProduct(u.components[0], rbm), mesh) / Integrate(material.rho() * InnerProduct(rbm, rbm), mesh)
        u_vec -= coeff * rbm.vec
        
    # Überschreibe mit projizierter Lösung
    u.components[0].vec.data = u_vec

    # Als VTK speichern
    save_ngsolve_result_as_vtk(path_info.verzeichnis+"/"+path_info.dateiname+"_modal_lagrange_multiplier_rbm.vtk", mesh, u.components[0], f)
