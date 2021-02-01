class LaplacianSolver:
    def __init__(self, params, xp):
        self._xp = xp
        self._params = params

        p = self._params

        n = xp.concatenate((xp.arange(0, p.nn+1),  xp.arange(-p.nn, 0)))
        m = xp.arange(0, p.nm)
        n, m = xp.meshgrid(n, m, indexing='ij')

        # Laplacian matrix
        self.lap = -((n*p.kn)**2 + (m*p.km)**2)
        self.lap[0,0] = 1

    def solve(self, in_arr, out_arr):
        # Avoid div by 0 errors
        self.lap[0,0] = 1.0
        out_arr[:] = -in_arr/self.lap
        self.lap[0,0] = 0.0
