import numpy as np
import pytest
from pytest import approx
from Parameters import Parameters
from SpectralTransformer import SpectralTransformer

@pytest.fixture
def parameters():
    PARAMS = {
        'nx': 4**2,
        'nz': 2**4
    }
    return Parameters(PARAMS, validate=False)

@pytest.fixture
def st(parameters):
    return SpectralTransformer(parameters, np)

@pytest.fixture
def spectral_array(parameters):
    p = parameters
    spectral = np.zeros((p.nn*2+1, p.nm), dtype=p.complex)
    return spectral

@pytest.fixture
def physical_array(parameters):
    p = parameters
    physical = np.zeros((p.nx, p.nz), dtype=p.float)
    return physical

@pytest.fixture
def arrays(spectral_array, physical_array):
    return spectral_array, physical_array

@pytest.fixture
def coordinates(parameters):
    p = parameters
    x = np.linspace(0, 1.0, p.nx, endpoint = False)
    z = np.linspace(0, 1.0, p.nz, endpoint = False)
    X, Z = np.meshgrid(x, z, indexing='ij')
    return X, Z


def test_to_physical(arrays, st, coordinates):
    # This relies on physical -> spectral working correctly
    spectral, physical = arrays
    X, Z = coordinates

    true_physical = np.cos(2*np.pi*X) + 2.0*np.sin(2*2*np.pi*Z)

    st.to_spectral(true_physical, spectral)
    st.to_physical(spectral, physical)

    assert physical == approx(true_physical)

def test_to_spectral(arrays, st, coordinates):
    spectral, physical = arrays
    X, Z = coordinates

    physical = np.cos(2*np.pi*X) + 2.0*np.sin(2*2*np.pi*Z)

    st.to_spectral(physical, spectral)

    # Note multiply spectral by 2 & -2. This is correct.
    assert 2*spectral[1, 0] == approx(1.0 + 0.0j)
    assert -2*spectral[0, 2] == approx(0.0 + 2.0j)
