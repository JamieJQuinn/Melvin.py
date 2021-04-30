import numpy as np
from utility import load_scipy_sparse, load_scipy_sparse_linalg

class LaplacianSolver:
    def __init__(self, params, xp, psi, array_factory=None):
        self._params = params
        self._xp = xp
        self._sparse = load_scipy_sparse(xp)
        self._linalg = load_scipy_sparse_linalg(xp)
        self._array_factory = array_factory

        p = self._params

        if p.is_fully_spectral():
            n, m = array_factory.make_mode_number_matrices()
            self.lap = -((n*np.abs(psi._ddx_factor))**2 + (m*np.abs(psi._ddz_factor))**2)
            self.solve = self._solve_fully_spectral
        else:
            offset = self._xp.array([-1, 0, 1])
            ones = self._xp.ones(p.nz)
            self.laps = [
                self._sparse.dia_matrix(
                    (
                        self._xp.array(
                            [1.0/p.dz**2*ones,
                             -((n_*np.abs(psi._ddx_factor))**2 + 2.0/p.dz**2)*ones,
                             1.0/p.dz**2*ones]
                        ), offset), 
                    shape=(p.nz, p.nz), dtype=p.float).tocsr()
                for n_ in range(p.nn)
            ]
            # PSI boundary conditions
            for lap in self.laps:
                lap[0,0] = 1.0
                lap[0,1] = 0.0
                lap[-1,-1] = 1.0
                lap[-1, -2] = 0.0
            self.solve = self._solve_fdm

    def _solve_fully_spectral(self, rhs, out=None):
        """Solves $\\omega = \\nabla^2 \\psi$ for $\\psi$ and sets this var to soln"""
        if out is None:
            out = self._array_factory.make_spectral()

        self.lap[0,0] = 1
        out = rhs/self.lap
        self.lap[0,0] = 0

        return out

    def _solve_fdm(self, rhs, out=None):
        if out is None:
            out = self._array_factory.make_spectral()

        p = self._params
        for n in range(p.nn):
            out[n,:] = self._linalg.spsolve(self.laps[n], rhs[n])

        return out
