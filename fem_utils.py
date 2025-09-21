from ngsolve import *

import numpy as np

def inner_product_M(rho, gf1, gf2):
    V = gf1.space
    mesh = V.mesh
    return Integrate(rho * InnerProduct(gf1, gf2), mesh)

def gram_schmidt_L2_ngsolve(rho, gfs):
    ortho_gfs = []
    V = gfs[0].space
    for i, gf in enumerate(gfs):
        # v = gf initial
        v_vec = gf.vec.Copy()
        for j, ogf in enumerate(ortho_gfs):
            gf_tmp = GridFunction(V)
            gf_tmp.vec.data = v_vec
            c = inner_product_M(rho, gf_tmp, ogf) / inner_product_M(rho, ogf, ogf)
            v_vec.data -= c * ogf.vec
        new_gf = GridFunction(V)
        new_gf.vec.data = v_vec
        norm = sqrt(inner_product_M(rho, new_gf, new_gf))
        new_gf.vec.data /= norm
        ortho_gfs.append(new_gf)
    return ortho_gfs

# Projiziere RBMs aus einem GridFunction heraus
def project_out_rbms(rho, gf, rbm_gfs_ortho):
    for rbm in rbm_gfs_ortho:
        num = inner_product_M(rho, gf, rbm)
        denom = inner_product_M(rho, rbm, rbm)
        gf.vec.data -= (num / denom) * rbm.vec
    return gf

def unit_check(rho_num, E_num, mesh_units='mm'):
    # assume rho in kg/mm^3, E in N/mm^2 if mesh in mm
    rho_SI = rho_num * 1e9
    E_SI   = E_num  * 1e6
    c = (E_SI / rho_SI)**0.5
    print(f"rho_SI = {rho_SI:.1f} kg/m^3, E_SI = {E_SI:.3e} Pa, c = {c:.1f} m/s")
    if 1e3 < c < 1e4:
        print("c in typical metal range -> units likely consistent.")
    else:
        print("c outside typical metal range -> check units of E and rho!")

def rigid_body_modes(V, rho):
    mesh = V.mesh
        # Berechne Massenmittelpunkt
    mass = Integrate(rho, mesh)
    center = (1.0 / mass) * Integrate(rho * CoefficientFunction((x, y, z)), mesh)
        # 6 RBMs in 3D: 3 Translationen, 3 Rotationen
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

    return rbm_gfs_ortho

def compute_condition_number(a, pre):
    lams = krylovspace.EigenValues_Preconditioner(mat=a.mat, pre=pre)
    lam_list = [abs(float(l)) for l in lams]

    filtered_lams = [l for l in lam_list if l > 1e-12]
    if not filtered_lams:
        kappa = float('inf')
    else:
        kappa = max(filtered_lams) / min(filtered_lams)
    
    return kappa