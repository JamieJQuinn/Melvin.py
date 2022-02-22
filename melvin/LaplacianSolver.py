import numpy as np
from melvin.utility import load_scipy_sparse, load_scipy_sparse_linalg
from melvin.BasisFunctions import calc_diff_factor


class LaplacianSolver:
    def __init__(
        self, params, xp, basis_fns, spatial_diff=None, array_factory=None
    ):
        self._params = params
        self._xp = xp
        self._sparse = load_scipy_sparse(xp)
        self._linalg = load_scipy_sparse_linalg(xp)
        self._array_factory = array_factory

        p = self._params

        if p.is_fully_spectral():
            self.lap = spatial_diff.calc_lap(basis_fns)
            self.solve = self._solve_fully_spectral
        else:
            offset = self._xp.array([-1, 0, 1])
            ones = self._xp.ones(p.nz)
            ddx_factor = calc_diff_factor(basis_fns[0], params.lx)
            self.laps = [
                self._sparse.dia_matrix(
                    (
                        self._xp.array(
                            [
                                1.0 / p.dz ** 2 * ones,
                                -(
                                    (n_ * np.abs(ddx_factor)) ** 2
                                    + 2.0 / p.dz ** 2
                                )
                                * ones,
                                1.0 / p.dz ** 2 * ones,
                            ]
                        ),
                        offset,
                    ),
                    shape=(p.nz, p.nz),
                    dtype=p.complex,
                ).tocsr()
                for n_ in range(p.nn)
            ]
            # boundary conditions TODO put in better location
            for lap in self.laps:
                lap[0, 0] = 1.0
                lap[0, 1] = 0.0
                lap[-1, -1] = 1.0
                lap[-1, -2] = 0.0
            self.solve = self._solve_fdm

    def _solve_fully_spectral(self, rhs, out=None):
        """Solves $\\omega = \\nabla^2 \\psi$
        for $\\psi$ and sets this var to soln"""
        if out is None:
            out = self._array_factory.make_spectral()

        self.lap[0, 0] = 1
        out[:] = rhs / self.lap
        self.lap[0, 0] = 0

        return out

    def _solve_fdm(self, rhs, out=None):
        if out is None:
            out = self._array_factory.make_spectral()

        p = self._params
        for n in range(p.nn):
            out[n, :] = self._linalg.spsolve(self.laps[n], rhs[n])

        return out
