class SpatialDifferentiator:

    def __init__(self, params, xp, n, m):
        self._xp = xp
        self._params = params

        self._n = n
        self._m = m

        if params.spatial_derivative_order == 2:
            self.pddx = self.__p_ddx_central2
            self.pddz = self.__p_ddz_central2
        elif params.spatial_derivative_order == 4:
            self.pddx = self.__p_ddx_central4
            self.pddz = self.__p_ddz_central4


    def sddx(self, var, out=None):
        if out is not None:
            out = 1j*self._params.kn*self._n*var
        else:
            return 1j*self._params.kn*self._n*var

    def sddz(self, var, out=None):
        if out is not None:
            out = 1j*self._params.km*self._m*var
        else:
            return 1j*self._params.km*self._m*var

    def __p_ddx_central2(self, var, out=None):
        if out is None:
            out = self._xp.zeros((self._params.nx, self._params.nz), dtype=self._params.float)

        dx = self._params.dx

        out[1:-1] = (var[2:] - var[:-2])/(2*dx)
        out[0] = (var[1] - var[-1])/(2*dx)
        out[-1] = (var[0] - var[-2])/(2*dx)

        return out

    def __p_ddz_central2(self, var, out=None):
        if out is None:
            out = self._xp.zeros((self._params.nx, self._params.nz), dtype=self._params.float)

        dz = self._params.dz

        out[:,1:-1] = (var[:,2:] - var[:,:-2])/(2*dz)
        out[:,0] = (var[:,1] - var[:,-1])/(2*dz)
        out[:,-1] = (var[:,0] - var[:,-2])/(2*dz)

        return out

    def __p_ddx_central4(self, var, out=None):
        if out is None:
            out = self._xp.zeros((self._params.nx, self._params.nz), dtype=self._params.float)

        dx = self._params.dx

        out[2:-2] = (-0.25*var[4:] + 2*var[3:-1] - 2*var[1:-3] + 0.25*var[:-4])/(3*dx)
        out[:2] = (-0.25*var[2:4] + 2*var[1:3] - 2*var.take((-1, 0), axis=0) + 0.25*var[-2:])/(3*dx)
        out[-2:] = (-0.25*var[:2] + 2*var.take((-1, 0), axis=0) - 2*var[-3:-1] + 0.25*var[-4:-2])/(3*dx)

        return out

    def __p_ddz_central4(self, var, out=None):
        if out is None:
            out = self._xp.zeros((self._params.nx, self._params.nz), dtype=self._params.float)

        dz = self._params.dz

        out[:,2:-2] = (-0.25*var[:,4:] + 2*var[:,3:-1] - 2*var[:,1:-3] + 0.25*var[:,:-4])/(3*dz)
        out[:,:2] = (-0.25*var[:,2:4] + 2*var[:,1:3] - 2*var.take((-1, 0), axis=1) + 0.25*var[:,-2:])/(3*dz)
        out[:,-2:] = (-0.25*var[:,:2] + 2*var.take((-1, 0), axis=1) - 2*var[:,-3:-1] + 0.25*var[:,-4:-2])/(3*dz)

        return out
