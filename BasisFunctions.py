from enum import Enum

class BasisFunctions(Enum):
    COMPLEX_EXP=1
    COSINE=2
    SINE=3
    FDM=4

def is_spectral(bs):
    """Returns whether a basis function is spectral"""
    return bs is BasisFunctions.COMPLEX_EXP \
            or bs is BasisFunctions.COSINE \
            or bs is BasisFunctions.SINE

def is_fully_spectral(bs1, bs2):
    """Returns whether all passed basis functions are spectral"""
    return is_spectral(bs1) and is_spectral(bs2)
