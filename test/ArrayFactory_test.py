import pytest
import numpy as np

from melvin import ArrayFactory


def test_fully_spectral_array_production(parameters):
    array_factory = ArrayFactory(parameters, np)

    spectral = array_factory.make_spectral()
    physical = array_factory.make_physical()

    assert spectral.shape == (2 * parameters.nn + 1, parameters.nm)


def test_fdm_array_production(fdm_parameters):
    array_factory = ArrayFactory(fdm_parameters, np)

    spectral = array_factory.make_spectral()
    physical = array_factory.make_physical()

    assert spectral.shape == (fdm_parameters.nn, fdm_parameters.nz)
    assert physical.shape == (fdm_parameters.nx, fdm_parameters.nz)
