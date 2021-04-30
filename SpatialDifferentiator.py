import numpy as np
from BasisFunctions import BasisFunctions

class SpatialDifferentiator:

    def __init__(self, params, xp, array_factory=None):
        self._xp = xp
        self._params = params

        self._x_periodic = (params.discretisation[0] == 'spectral')
        self._z_periodic = (params.discretisation[1] == 'spectral')

        self._n, self._m = array_factory.make_mode_number_matrices()

        if params.spatial_derivative_order == 2:
            self.pddx = self.__p_ddx_central2
            self.pddz = self.__p_ddz_central2
            self.pd2dz2 = self.__p_d2dz2_central2
        elif params.spatial_derivative_order == 4:
            self.pddx = self.__p_ddx_central4
            self.pddz = self.__p_ddz_central4

    def sddx(self, var, ddx_factor, out=None):
        if out is not None:
            out[:] = ddx_factor*self._n*var
        else:
            return ddx_factor*self._n*var

    def sddz(self, var, ddz_factor, out=None):
        if out is not None:
            out[:] = ddz_factor*self._m*var
        else:
            return ddz_factor*self._m*var

    def sd2dx2(self, var, ddx_factor, out=None):
        if out is not None:
            out[:] = -(np.abs(ddx_factor)*self._n)**2*var
        else:
            return -(np.abs(ddx_factor)*self._n)**2*var

    def sd2dz2(self, var, ddz_factor, out=None):
        if out is not None:
            out[:] = -(np.abs(ddz_factor)*self._m)**2*var
        else:
            return -(np.abs(ddz_factor)*self._m)**2*var

    def __p_ddx_central2(self, var, out=None):
        if out is None:
            out = self._xp.zeros((self._params.nx, self._params.nz), dtype=self._params.float)

        dx = self._params.dx

        out[1:-1] = (var[2:] - var[:-2])/(2*dx)
        if self._x_periodic:
            out[0] = (var[1] - var[-1])/(2*dx)
            out[-1] = (var[0] - var[-2])/(2*dx)

        return out

    def __p_ddz_central2(self, var, out=None):
        if out is None:
            out = self._xp.zeros((self._params.nx, self._params.nz), dtype=self._params.float)

        dz = self._params.dz

        out[:,1:-1] = (var[:,2:] - var[:,:-2])/(2*dz)
        if self._z_periodic:
            out[:,0] = (var[:,1] - var[:,-1])/(2*dz)
            out[:,-1] = (var[:,0] - var[:,-2])/(2*dz)

        return out

    def __p_d2dz2_central2(self, var, out=None):
        if out is None:
            out = self._xp.zeros_like(var)

        dz = self._params.dz

        out[:,1:-1] = (var[:,2:] - 2*var[:,1:-1] + var[:,:-2])/(dz**2)

        return out

    def __p_ddx_central4(self, var, out=None):
        if out is None:
            out = self._xp.zeros((self._params.nx, self._params.nz), dtype=self._params.float)

        dx = self._params.dx

        out[2:-2] = (-0.25*var[4:] + 2*var[3:-1] - 2*var[1:-3] + 0.25*var[:-4])/(3*dx)
        if self._x_periodic:
            out[:2] = (-0.25*var[2:4] + 2*var[1:3] - 2*var.take((-1, 0), axis=0) + 0.25*var[-2:])/(3*dx)
            out[-2:] = (-0.25*var[:2] + 2*var.take((-1, 0), axis=0) - 2*var[-3:-1] + 0.25*var[-4:-2])/(3*dx)

        return out

    def __p_ddz_central4(self, var, out=None):
        if out is None:
            out = self._xp.zeros((self._params.nx, self._params.nz), dtype=self._params.float)

        dz = self._params.dz

        out[:,2:-2] = (-0.25*var[:,4:] + 2*var[:,3:-1] - 2*var[:,1:-3] + 0.25*var[:,:-4])/(3*dz)
        if self._z_periodic:
            out[:,:2] = (-0.25*var[:,2:4] + 2*var[:,1:3] - 2*var.take((-1, 0), axis=1) + 0.25*var[:,-2:])/(3*dz)
            out[:,-2:] = (-0.25*var[:,:2] + 2*var.take((-1, 0), axis=1) - 2*var[:,-3:-1] + 0.25*var[:,-4:-2])/(3*dz)

        return out
