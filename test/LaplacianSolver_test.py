import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from melvin import Variable, LaplacianSolver, ArrayFactory, BasisFunctions
from melvin.utility import load_scipy_sparse


def test_fdm_laplacian_solver(fdm_parameters):
    p = fdm_parameters
    array_factory = ArrayFactory(p, np)

    basis_fns = [BasisFunctions.COMPLEX_EXP, BasisFunctions.COMPLEX_EXP]

    psi = Variable(p, np, array_factory=array_factory, basis_functions=basis_fns)

    laplacian_solver = LaplacianSolver(p, np, psi, array_factory=array_factory)

    true_soln = array_factory.make_spectral()
    rhs = array_factory.make_spectral()

    true_soln[:, :] = 1.0

    for n in range(p.nn):
        rhs[n, :] = laplacian_solver.laps[n] @ true_soln[n, :]

    soln = laplacian_solver.solve(rhs)

    assert_array_almost_equal(soln, true_soln)
