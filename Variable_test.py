import numpy as np
import pytest
import time

from pytest import approx
from Parameters import Parameters
from SpectralTransformer import SpectralTransformer
from SpatialDifferentiator import SpatialDifferentiator
from Variable import Variable

@pytest.fixture
def parameters():
    PARAMS = {
        'nx': 4**4,
        'nz': 2**8,
        'lx': 1.0,
        'lz': 1.0,
        'final_time': 1.0
    }
    return Parameters(PARAMS, validate=False)

@pytest.fixture
def st(parameters):
    return SpectralTransformer(parameters, np)

@pytest.fixture
def sd(parameters):
    return SpatialDifferentiator(parameters, np, None, None)

def test_spatial_derivatives(parameters, st, sd):
    p = parameters
    x = np.linspace(0, p.lx, p.nx, endpoint=False)
    z = np.linspace(0, p.lz, p.nz, endpoint=False)
    X, Z = np.meshgrid(x, z, indexing='ij')

    var = Variable(parameters, np, st=st, sd=sd)
    var.setp(np.cos(2*np.pi*X)*np.cos(2*np.pi*Z))

    dvardx = var.pddx()

    dvardx_true = -2*np.pi*np.sin(2*np.pi*x)
    dvardx_test = dvardx[:,0]

    assert dvardx_test == approx(dvardx_true, rel=1e-3)

    dvardz = var.pddz()

    dvardz_true = -2*np.pi*np.sin(2*np.pi*z)
    dvardz_test = dvardz[0,:]

    assert dvardz_test == approx(dvardz_true, rel=1e-3)
