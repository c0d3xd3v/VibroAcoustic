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
        pre = ngsolve.Preconditioner(a, type="direct", inverse = "masterinverse")

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


def solve_elasticity_system(_material, _fes, count=20, filter_modes=True):
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

    _a, _b, pre = fe_preconditioning(_fes, _a, _b)

    u = ngsolve.GridFunction(_fes, multidim=count)
    lams = ngsolve.ArnoldiSolver(_a.mat, _b.mat, _fes.FreeDofs(), list(u.vecs), 4000, inverse="pardiso")
    
    # compute the frequencies from squared natural frequencies
    # lams = omega**2
    # f = sqrt(lams)/(2.0 * pi)
    f = np.sqrt(lams)/(2.0 * np.pi)

    filtered_f = []
    filtered_u = ngsolve.GridFunction(u.space)
    
    for i, f_ in enumerate(f):
        if filter_modes:
            vec_norm = np.linalg.norm(u.vecs[i])
            if not np.isnan(f_) and f_ > 1.0 and vec_norm != 0.0:
                filtered_f.append(np.round(f_, decimals=2))
                filtered_u.AddMultiDimComponent(u.vecs[i])
        else:
            filtered_f.append(np.round(f_, decimals=2))
            filtered_u.AddMultiDimComponent(u.vecs[i])

    return filtered_u, filtered_f
