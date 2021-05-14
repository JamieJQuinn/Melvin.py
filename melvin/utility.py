import numpy as np
import scipy.sparse
import scipy.sparse.linalg


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

