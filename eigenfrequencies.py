import os
import sys

import ngsolve
import numpy as np


def build_simple_solid_fes(mesh):
    fes:ngsolve.VectorH1 = ngsolve.VectorH1(mesh, order=2, dirichlet=[], complex=True)
    return fes


def fe_preconditioning(fes, a, b, precond=None):
    # bddc, h1amg, multigrid, local, direct
    # precond = 'h1amg'
    pre = None
    if precond == 'identity':
        pre = ngsolve.IdentityMatrix(fes.ndof, complex=True)
    else:
        #jac = a.mat.CreateBlockSmoother(solid_fes.CreateSmoothingBlocks())
        #preJpoint = a.mat.CreateSmoother(solid_fes.FreeDofs())
        pre = ngsolve.Preconditioner(a, type="direct", inverse = "pardiso")

    a.Assemble()
    b.Assemble()

    return a, b, pre


def compute_condition_number(a, pre):
    lams = ngsolve.krylovspace.EigenValues_Preconditioner(mat=a.mat, pre=pre)
    l0 = min(lams)
    l1 = max(lams)
    kapa = -1.
    if l0 != 0.:
        kapa = l1/l0
    return kapa


def solve_elasticity_system(_material, _fes, count=12):
    _u, _v = _fes.TrialFunction(), _fes.TestFunction()
    #_a = ngsolve.BilinearForm(_fes, symmetric=True, eliminate_internal=True)
    _a = ngsolve.BilinearForm(_fes, symmetric=True)
    _a += ngsolve.SymbolicBFI(2 * _material.mu
                              * ngsolve.InnerProduct(1.0 / 2.0 * (ngsolve.grad(_u) + ngsolve.grad(_u).trans),
                                                     1.0 / 2.0 * (ngsolve.grad(_v) + ngsolve.grad(_v).trans))
                              + _material.lam * ngsolve.div(_u) * ngsolve.div(_v))
    #_b = ngsolve.BilinearForm(_fes, symmetric=True, eliminate_internal=True)
    _b = ngsolve.BilinearForm(_fes, symmetric=True)
    _b += ngsolve.SymbolicBFI(_material.rho * _u * _v)

    pre = ngsolve.Preconditioner(_a, type="h1amg")  # oder "direct", etc.

    _a.Assemble()
    _b.Assemble()

    pre.Update()

    lams = ngsolve.krylovspace.EigenValues_Preconditioner(mat=_a.mat, pre=pre)

    lam_list = [abs(float(l)) for l in lams]

    filtered_lams = [l for l in lam_list if l > 1e-12]
    if not filtered_lams:
        kappa = float('inf')
    else:
        kappa = max(filtered_lams) / min(filtered_lams)

    print(f"Konditionszahl: {kappa:.2e}")


    u = ngsolve.GridFunction(_fes, multidim=count)
    lams = ngsolve.ArnoldiSolver(_a.mat, _b.mat, _fes.FreeDofs(), list(u.vecs), 1000, inverse="pardiso")
    
    # compute the frequencies from squared natural frequencies
    # lams = omega**2
    # f = sqrt(lams)/(2.0 * pi)
    f = np.sqrt(lams)/(2.0 * np.pi)

    return u, f
