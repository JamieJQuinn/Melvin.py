import pytest
import numpy as np

from melvin import (
    Parameters,
    SpectralTransformer,
    SpatialDifferentiator,
    ArrayFactory,
)


@pytest.fixture
def parameters():
    PARAMS = {
        "nx": 4 ** 4,
        "nz": 2 ** 8,
        "lx": 1.0,
        "lz": 1.0,
        "final_time": 1.0,
    }
    return Parameters(PARAMS, validate=False)


@pytest.fixture
def fdm_parameters():
    PARAMS = {
        "nx": 4 ** 4,
        "nz": 2 ** 8,
        "lx": 1.0,
        "lz": 1.0,
        "final_time": 1.0,
        "discretisation": ["spectral", "fdm"],
    }
    return Parameters(PARAMS, validate=False)


@pytest.fixture
def array_factory(parameters):
    return ArrayFactory(parameters, np)


@pytest.fixture
def st(parameters, array_factory):
    return SpectralTransformer(parameters, np, array_factory)


@pytest.fixture
def sd(parameters, array_factory):
    return SpatialDifferentiator(parameters, np, array_factory)


@pytest.fixture
def arrays(array_factory):
    return array_factory.make_spectral(), array_factory.make_physical()
