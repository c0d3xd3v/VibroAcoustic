import os

import ngsolve
from ngsolve import *

from fileio import load_volume_mesh
from fileio import save_ngsolve_result_as_vtk
from fileio import split_path_informations

from fem_utils import unit_check
from fem_utils import gram_schmidt_L2_ngsolve

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

    rho = 2.5355e-6   # kg/mm³
    E   =  6.9e7     # N/mm²

    unit_check(rho, E, "mm")

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

    print(f"dx: ({xmax - xmin})")
    print(f"dy: ({ymax - ymin})")
    print(f"dz: ({zmax - zmin})")

    # Tensoren
    def eps(w): return 0.5*(grad(w)+grad(w).trans)
    def sigma(w): return lam*div(w)*Id(3) + 2*mu_val*eps(w)

    # Berechne Massenmittelpunkt
    mass = Integrate(rho, mesh)
    center = (1.0 / mass) * Integrate(rho * CoefficientFunction((x, y, z)), mesh)

    print("Masse des Körpers:", mass, "kg")

    # 6 RBMs in 3D: 3 Translationen, 3 Rotationen
    from ngsolve import x, y, z
    cx, cy, cz = center[0], center[1], center[2] # Mittelpunkt des Balkens
    rbms = [
        CoefficientFunction((1, 0, 0)),  # Translation x
        CoefficientFunction((0, 1, 0)),  # Translation y
        CoefficientFunction((0, 0, 1)),  # Translation z
        CoefficientFunction((0, z-cz, -(y-cy))),  # Rotation x-Achse
        CoefficientFunction((-(z-cz), 0, x-cx)),  # Rotation y-Achse
        CoefficientFunction((y-cy, -(x-cx), 0))   # Rotation z-Achse
    ]

    # orthogonalisieren
    rbm_gfs = [GridFunction(V) for _ in rbms]
    for gf, expr in zip(rbm_gfs, rbms):
        gf.Set(expr)
    rbm_gfs_ortho = gram_schmidt_L2_ngsolve(rho, rbm_gfs)

    # teste orthogonalität
    n = len(rbm_gfs)
    G = np.zeros((n,n), dtype=complex)
    for i in range(n):
        for j in range(n):
            integrand = rho * InnerProduct(rbm_gfs_ortho[i], rbm_gfs_ortho[j])
            G[i,j] = Integrate(integrand, mesh)
    print("orthogonalität :", np.linalg.norm(G - np.eye(6)))

    # Bilinearform
    a = BilinearForm(X, symmetric=True)
    a += SymbolicBFI(InnerProduct(sigma(u), eps(v)))

    rbm_coeffs = [CoefficientFunction(rbm_gf) for rbm_gf in rbm_gfs_ortho]
    for i in range(len(rbm_gfs_ortho)):
        a += SymbolicBFI(InnerProduct(rbm_coeffs[i], v) * lambdas[i])
        a += SymbolicBFI(InnerProduct(rbm_coeffs[i], u) * mus[i])

    b = BilinearForm(X, symmetric=True)
    b.components[0] += SymbolicBFI(rho*InnerProduct(V.TrialFunction(), V.TestFunction()))
    for i in range(1, len(rbm_gfs_ortho)+1):
        b.components[i] += SymbolicBFI(rho*CoefficientFunction(0.0))

    pre = ngsolve.Preconditioner(a, type="direct")  # oder "bddc, h1amg, direct", etc.

    a.Assemble()
    b.Assemble()

    pre.Update()

    lams = ngsolve.krylovspace.EigenValues_Preconditioner(mat=a.mat, pre=pre)
    lam_list = [abs(float(l)) for l in lams]

    filtered_lams = [l for l in lam_list if l > 1e-12]
    if not filtered_lams:
        kappa = float('inf')
    else:
        kappa = max(filtered_lams) / min(filtered_lams)

    print(f"Konditionszahl(pre x mat) : {kappa:.2e}")

    # Lösung
    lams, u = solve_scalar_eigen_problem_a(rho, X, a.mat, b.mat, rbm_gfs_ortho, 12)
    f = np.sqrt(lams) / (2.0 * np.pi)

    for i, rbm in enumerate(rbm_gfs_ortho):
        val = Integrate(rho * InnerProduct(rbm, u.components[0]), mesh)
        print(f"Lagrange-Kopplung {i}: {val}")

    u_vec = u.components[0].vec.Copy()

    for rbm in rbm_gfs_ortho:
        coeff = Integrate(rho * InnerProduct(u.components[0], rbm), mesh) / Integrate(rho * InnerProduct(rbm, rbm), mesh)
        u_vec -= coeff * rbm.vec
        
    # Überschreibe mit projizierter Lösung
    u.components[0].vec.data = u_vec

    # Als VTK speichern
    save_ngsolve_result_as_vtk(path_info.dateiname+"_modal_lagrange_multiplier_rbm.vtk", mesh, u.components[0], f)
