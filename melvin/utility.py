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
