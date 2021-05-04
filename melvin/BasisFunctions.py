import numpy as np
from enum import IntEnum


class BasisFunctions(IntEnum):
    COMPLEX_EXP = 0
    COSINE = 1
    SINE = 2
    FDM = 3


def is_spectral(bs):
    """Returns whether a basis function is spectral"""
    return (
        bs is BasisFunctions.COMPLEX_EXP
        or bs is BasisFunctions.COSINE
        or bs is BasisFunctions.SINE
    )


def is_fully_spectral(bs1, bs2):
    """Returns whether all passed basis functions are spectral"""
    return is_spectral(bs1) and is_spectral(bs2)


def calc_diff_wavelength(basis_fn):
    """Returns wavelength of basis function
    for use in first order differentiation"""
    if basis_fn is BasisFunctions.COMPLEX_EXP:
        return 1j * 2 * np.pi
    elif basis_fn is BasisFunctions.SINE:
        return np.pi
    elif basis_fn is BasisFunctions.COSINE:
        return -np.pi
    else:
        return 0


def calc_diff2_wavelength(basis_fn):
    """Returns wavelength of basis function
    for use in 2nd order differentiation"""
    return -np.abs(calc_diff_wavelength(basis_fn)) ** 2


def calc_diff_factor(basis_fn, length):
    return calc_diff_wavelength(basis_fn) / length


def gen_diff_factors(length):
    return [
        calc_diff_wavelength(basis_fn) / length for basis_fn in BasisFunctions
    ]


def gen_diff2_factors(length):
    return [
        calc_diff2_wavelength(basis_fn) / length ** 2
        for basis_fn in BasisFunctions
    ]
