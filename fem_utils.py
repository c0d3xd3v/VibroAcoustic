from ngsolve import *


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

