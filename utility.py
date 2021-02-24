import numpy as np

def sech(x):
    return 1.0/np.cosh(x)

def set_numpy_module(PARAMS):
    if PARAMS['cuda_enabled']:
        return cupy
    else:
        return numpy
