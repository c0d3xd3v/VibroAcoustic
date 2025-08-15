from ngsolve import *

from fem_utils import project_out_rbms

from numpy import  random


# Eigenwertproblem mit Arnoldi
def solve_scalar_eigen_problem_a(rho, V, a_mat, m_mat, rbm_gfs_ortho, num=10):
    gf = GridFunction(V, multidim=num)
    # Startvektor orthogonalisieren gegen Starkörperbewegungen
    # => suche im Orthogonalen Komplement der Starkörperbewegungen starten,
    # evtl. numerische probleme ? noch klären.
    for i in range(num):
        gf.vecs[i].FV().NumPy()[:] = random.rand(V.ndof)
        tmp_gf = GridFunction(V)
        tmp_gf.vec.data = gf.vecs[i]
        tmp_gf = project_out_rbms(rho, tmp_gf.components[0], rbm_gfs_ortho)
        gf.components[0].vecs[i].data = tmp_gf.vec
    fdofs = V.FreeDofs()
    lams = ArnoldiSolver(a_mat, m_mat, fdofs, list(gf.vecs), 100, inverse="pardiso")
    return lams, gf


def solve_scalar_eigen_problem_p(V, a_mat, m_mat, rbm_gfs_ortho, num=12):
    # PARDISO als inverser Operator
    inv = a_mat.Inverse(inverse="pardiso")  # <--- HIER
    evals, evecs = solvers.PINVIT(a_mat, m_mat, pre=inv, num=num, maxit=5)

    gf = GridFunction(V, multidim=num)
    for i in range(num):
        gf.vecs[i].data[:] = evecs[i]

    return evals, gf
