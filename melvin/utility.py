import numpy as np
from numpy.random import default_rng
import scipy.sparse
import scipy.sparse.linalg

from melvin import BasisFunctions


def sech(x):
    return 1.0 / np.cosh(x)


def load_scipy_sparse(xp):
    if xp.__name__ == "numpy":
        return scipy.sparse
    elif xp.__name__ == "cupy":
        import cupyx

        return cupyx.scipy.sparse


def load_scipy_sparse_linalg(xp):
    if xp.__name__ == "numpy":
        return scipy.sparse.linalg
    elif xp.__name__ == "cupy":
        import cupyx.scipy.sparse.linalg

        return cupyx.scipy.sparse.linalg


def init_var_with_noise(var, epsilon, seed=0):
    rng = default_rng(seed)

    var_shape = var.getp().shape

    data_p = np.zeros(var_shape)
    data_p += epsilon * (2 * rng.random(var_shape) - 1.0)

    var.load(data_p, is_physical=True)


def calc_kinetic_energy(ux, uz, xp, params):
    """
    Calculates the kinetic energy from velocity

    Args:
        ux (np.ndarray): x-velocity
        uz (np.ndarray): z-velocity

        xp (module): numpy module
        params (Parameters): parameters

    Returns:
        float: kinetic energy
    """
    nx, nz = params.nx, params.nz
    ke = uz.getp() ** 2 + ux.getp() ** 2
    total_ke = 0.5 * xp.sum(ke) / (nx * nz)
    return total_ke


def calc_velocity_from_vorticity(
    vorticity, streamfunction, ux, uz, laplacian_solver
):
    laplacian_solver.solve(-vorticity.gets(), out=streamfunction._sdata)

    if streamfunction._basis_functions[1] is BasisFunctions.FDM:
        streamfunction.to_physical()
        ux.setp(-streamfunction.pddz())
    else:
        ux[:] = -streamfunction.sddz()
        ux.to_physical()

    if streamfunction._basis_functions[0] is BasisFunctions.FDM:
        streamfunction.to_physical()
        uz.setp(streamfunction.pddx())
    else:
        uz[:] = streamfunction.sddx()
        uz.to_physical()
