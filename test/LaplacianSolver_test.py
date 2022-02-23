import numpy as np
import scipy
from numpy.testing import assert_array_almost_equal
from functools import partial

from melvin import Variable, LaplacianSolver, ArrayFactory, BasisFunctions


# def test_linear_operator(fdm_parameters):
    # N = 100
    # dx = 1./(N-1)

    # b = np.linspace(0, 1., N)
    # b[0] = -1./dx**2
    # b[-1] = -1./dx**2

    # buffer = np.zeros_like(b)

    # def ddx(v, dx, out=None):
        # if out is None:
            # out = np.zeros_like(v)
        # out[1:-1] = (v[:-2] + -2.*v[1:-1] + v[2:])/(dx**2)
        # return out

    # # Create linear operator
    # linop = scipy.sparse.linalg.LinearOperator(
        # (N, N),
        # partial(ddx, dx=dx, out=buffer)
    # )

    # x = scipy.sparse.linalg.cg(linop, b)

    # print(x)
    # assert(False)


def test_fdm_laplacian_solver(fdm_parameters):
    p = fdm_parameters
    array_factory = ArrayFactory(p, np)

    basis_fns = [BasisFunctions.COMPLEX_EXP, BasisFunctions.COMPLEX_EXP]

    psi = Variable(
        p, np, array_factory=array_factory, basis_functions=basis_fns
    )

    laplacian_solver = LaplacianSolver(p, np, psi, array_factory=array_factory)

    true_soln = array_factory.make_spectral()
    rhs = array_factory.make_spectral()

    true_soln[:, :] = 1.0

    for n in range(p.nn):
        rhs[n, :] = laplacian_solver.laps[n] @ true_soln[n, :]

    soln = laplacian_solver.solve(rhs)

    assert_array_almost_equal(soln, true_soln)
