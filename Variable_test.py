import numpy as np
import pytest
from pytest import approx
from Parameters import Parameters
from SpectralTransformer import SpectralTransformer
from Variable import Variable

@pytest.fixture
def parameters():
    PARAMS = {
        'nx': 4**2,
        'nz': 2**4,
        'lx': 1.0,
        'lz': 1.0,
        'dt': 1.0
    }
    return Parameters(PARAMS)

@pytest.fixture
def st(parameters):
    return SpectralTransformer(parameters, np)

def test_ddx(parameters, st):
    p = parameters
    x = np.linspace(0, p.lx, p.nx, endpoint=False)
    z = np.linspace(0, p.lz, p.nz, endpoint=False)
    X, Z = np.meshgrid(x, z, indexing='ij')

    var = Variable(parameters, np, st)
    var.set_internal(np.sin(2*np.pi*X)*np.sin(2*np.pi*Z))

    dvardx = np.zeros((p.nx, p.nz+2))
    var.ddx(dvardx)

    dvardx_true = 2*np.pi*np.cos(2*np.pi*x)
    dvardx_test = dvardx[:,int(p.nz/4)]

    assert dvardx_test == approx(dvardx_true)

