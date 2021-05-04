from melvin.BasisFunctions import (
    gen_diff_factors,
    gen_diff2_factors,
)


class SpatialDifferentiator:
    def __init__(self, params, xp, array_factory=None):
        self._xp = xp
        self._params = params

        self._x_periodic = params.discretisation[0] == "spectral"
        self._z_periodic = params.discretisation[1] == "spectral"

        self._n, self._m = array_factory.make_mode_number_matrices()

        if params.spatial_derivative_order == 2:
            self.pddx = self.__p_ddx_central2
            self.pddz = self.__p_ddz_central2
            self.pd2dz2 = self.__p_d2dz2_central2
        elif params.spatial_derivative_order == 4:
            self.pddx = self.__p_ddx_central4
            self.pddz = self.__p_ddz_central4

        if params.discretisation[0] == "fdm":
            # self.sddx = self.pddx
            # self.sd2dx2 = self.pd2dx2
            raise NotImplementedError(
                "Finite difference not implemented in x direction"
            )
        elif params.discretisation[0] == "spectral":
            self.sddx = self.__s_ddx
            self.sd2dx2 = self.__s_d2dx2

        if params.discretisation[1] == "fdm":
            self.sddz = lambda var, bs: self.pddz(
                var
            )  # This strips out the extra paramters not required by pddx
            self.sd2dz2 = lambda var, bs: self.pd2dz2(var)
        elif params.discretisation[1] == "spectral":
            self.sddz = self.__s_ddz
            self.sd2dz2 = self.__s_d2dz2

        self._diffx_factors = gen_diff_factors(params.lx)
        self._diffz_factors = gen_diff_factors(params.lz)
        self._diff2x_factors = gen_diff2_factors(params.lx)
        self._diff2z_factors = gen_diff2_factors(params.lz)

    def __s_ddx(self, var, basis_fn):
        """Calculate first order derivative wrt x in spectral space"""
        diff_factor = self._diffx_factors[basis_fn]
        return diff_factor * self._n * var

    def __s_ddz(self, var, basis_fn):
        """Calculate first order derivative wrt z in spectral space"""
        diff_factor = self._diffz_factors[basis_fn]
        return diff_factor * self._m * var

    def __s_d2dx2(self, var, basis_fn):
        """Calculate second order derivative wrt x in spectral space"""
        diff2_factor = self._diff2x_factors[basis_fn]
        return diff2_factor * self._n ** 2 * var

    def __s_d2dz2(self, var, basis_fn):
        """Calculate second order derivative wrt z in spectral space"""
        diff2_factor = self._diff2z_factors[basis_fn]
        return diff2_factor * self._m ** 2 * var

    def calc_lap(self, basis_fns):
        return (
            self._diff2x_factors[basis_fns[0]] * self._n ** 2
            + self._diff2z_factors[basis_fns[1]] * self._m ** 2
        )

    def __p_ddx_central2(self, var, out=None):
        if out is None:
            out = self._xp.zeros(
                (self._params.nx, self._params.nz), dtype=self._params.float
            )

        dx = self._params.dx

        out[1:-1] = (var[2:] - var[:-2]) / (2 * dx)
        if self._x_periodic:
            out[0] = (var[1] - var[-1]) / (2 * dx)
            out[-1] = (var[0] - var[-2]) / (2 * dx)

        return out

    def __p_ddz_central2(self, var, out=None):
        if out is None:
            out = self._xp.zeros(
                (self._params.nx, self._params.nz), dtype=self._params.float
            )

        dz = self._params.dz

        out[:, 1:-1] = (var[:, 2:] - var[:, :-2]) / (2 * dz)
        if self._z_periodic:
            out[:, 0] = (var[:, 1] - var[:, -1]) / (2 * dz)
            out[:, -1] = (var[:, 0] - var[:, -2]) / (2 * dz)

        return out

    def __p_d2dz2_central2(self, var, out=None):
        if out is None:
            out = self._xp.zeros_like(var)

        dz = self._params.dz

        out[:, 1:-1] = (var[:, 2:] - 2 * var[:, 1:-1] + var[:, :-2]) / (
            dz ** 2
        )

        return out

    def __p_ddx_central4(self, var, out=None):
        if out is None:
            out = self._xp.zeros(
                (self._params.nx, self._params.nz), dtype=self._params.float
            )

        dx = self._params.dx

        out[2:-2] = (
            -0.25 * var[4:] + 2 * var[3:-1] - 2 * var[1:-3] + 0.25 * var[:-4]
        ) / (3 * dx)
        if self._x_periodic:
            out[:2] = (
                -0.25 * var[2:4]
                + 2 * var[1:3]
                - 2 * var.take((-1, 0), axis=0)
                + 0.25 * var[-2:]
            ) / (3 * dx)
            out[-2:] = (
                -0.25 * var[:2]
                + 2 * var.take((-1, 0), axis=0)
                - 2 * var[-3:-1]
                + 0.25 * var[-4:-2]
            ) / (3 * dx)

        return out

    def __p_ddz_central4(self, var, out=None):
        if out is None:
            out = self._xp.zeros(
                (self._params.nx, self._params.nz), dtype=self._params.float
            )

        dz = self._params.dz

        out[:, 2:-2] = (
            -0.25 * var[:, 4:]
            + 2 * var[:, 3:-1]
            - 2 * var[:, 1:-3]
            + 0.25 * var[:, :-4]
        ) / (3 * dz)
        if self._z_periodic:
            out[:, :2] = (
                -0.25 * var[:, 2:4]
                + 2 * var[:, 1:3]
                - 2 * var.take((-1, 0), axis=1)
                + 0.25 * var[:, -2:]
            ) / (3 * dz)
            out[:, -2:] = (
                -0.25 * var[:, :2]
                + 2 * var.take((-1, 0), axis=1)
                - 2 * var[:, -3:-1]
                + 0.25 * var[:, -4:-2]
            ) / (3 * dz)

        return out
