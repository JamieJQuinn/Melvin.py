import numpy as np
import pytest
from pytest import approx
from numpy.testing import assert_array_almost_equal

from melvin import BasisFunctions, ArrayFactory, SpectralTransformer


@pytest.fixture
def periodic_coordinates(parameters):
    p = parameters
    x = np.linspace(0, 1.0, p.nx, endpoint=False)
    z = np.linspace(0, 1.0, p.nz, endpoint=False)
    X, Z = np.meshgrid(x, z, indexing="ij")
    return X, Z


@pytest.fixture
def coordinates(parameters):
    p = parameters
    x = np.linspace(0, 1.0, p.nx)
    z = np.linspace(0, 1.0, p.nz)
    X, Z = np.meshgrid(x, z, indexing="ij")
    return X, Z


def test_transform_periodic(arrays, st, periodic_coordinates):
    spectral, physical = arrays
    X, Z = periodic_coordinates

    true_physical = np.cos(2 * np.pi * X) + 2.0 * np.sin(2 * 2 * np.pi * Z)

    st.to_spectral(
        true_physical,
        spectral,
        basis_functions=[
            BasisFunctions.COMPLEX_EXP,
            BasisFunctions.COMPLEX_EXP,
        ],
    )

    true_spectral = np.zeros_like(spectral)

    # These values are those in front of cosine divided by two
    true_spectral[1, 0] = 1.0 / 2.0
    true_spectral[-1, 0] = 1.0 / 2.0  # complex conjugate is also calculated

    # Sine coefficients are complex, negative and also divided by two
    true_spectral[0, 2] = 2.0j / -2.0

    assert_array_almost_equal(spectral, true_spectral)

    st.to_physical(
        spectral,
        physical,
        basis_functions=[
            BasisFunctions.COMPLEX_EXP,
            BasisFunctions.COMPLEX_EXP,
        ],
    )
    assert_array_almost_equal(physical, true_physical)


def test_transform_cosine_x_periodic_z(arrays, st, parameters):
    spectral, physical = arrays

    p = parameters
    x = np.linspace(0, 1.0, p.nx)
    z = np.linspace(0, 1.0, p.nz, endpoint=False)
    X, Z = np.meshgrid(x, z, indexing="ij")

    true_physical = np.cos(np.pi * X) + 2.0 * np.cos(2 * np.pi * X)

    st.to_spectral(
        true_physical,
        spectral,
        basis_functions=[BasisFunctions.COSINE, BasisFunctions.COMPLEX_EXP],
    )

    true_spectral = np.zeros_like(spectral)
    true_spectral[1, 0] = 1.0
    true_spectral[2, 0] = 2.0
    true_spectral[-1, 0] = 1.0
    true_spectral[-2, 0] = 2.0

    assert_array_almost_equal(spectral, true_spectral)

    st.to_physical(
        spectral,
        physical,
        basis_functions=[BasisFunctions.COSINE, BasisFunctions.COMPLEX_EXP],
    )
    assert_array_almost_equal(physical, true_physical)


def test_transform_periodic_x_cosine_z(arrays, st, parameters):
    spectral, physical = arrays

    p = parameters
    x = np.linspace(0, 1.0, p.nx, endpoint=False)
    z = np.linspace(0, 1.0, p.nz)
    X, Z = np.meshgrid(x, z, indexing="ij")

    true_physical = np.cos(np.pi * Z) + 2.0 * np.cos(2 * np.pi * Z)

    st.to_spectral(
        true_physical,
        spectral,
        basis_functions=[BasisFunctions.COMPLEX_EXP, BasisFunctions.COSINE],
    )

    true_spectral = np.zeros_like(spectral)
    true_spectral[0, 1] = 1.0
    true_spectral[0, 2] = 2.0

    assert_array_almost_equal(spectral, true_spectral)

    st.to_physical(
        spectral,
        physical,
        basis_functions=[BasisFunctions.COMPLEX_EXP, BasisFunctions.COSINE],
    )
    assert_array_almost_equal(physical, true_physical)


def test_transform_cosine_both(arrays, st, parameters):
    spectral, physical = arrays

    p = parameters
    x = np.linspace(0, 1.0, p.nx, endpoint=True)
    z = np.linspace(0, 1.0, p.nz, endpoint=True)
    X, Z = np.meshgrid(x, z, indexing="ij")

    true_physical = (
        np.cos(np.pi * Z)
        + 2.0 * np.cos(2 * np.pi * Z)
        + np.cos(np.pi * X)
        + 2.0 * np.cos(2 * np.pi * X) * np.cos(1 * np.pi * Z)
    )

    st.to_spectral(
        true_physical,
        spectral,
        basis_functions=[BasisFunctions.COSINE, BasisFunctions.COSINE],
    )

    true_spectral = np.zeros_like(spectral)
    true_spectral[0, 1] = 1.0
    true_spectral[0, 2] = 2.0

    true_spectral[1, 0] = 1.0
    true_spectral[2, 1] = 2.0
    true_spectral[-1, 0] = 1.0
    true_spectral[-2, 1] = 2.0

    # np.set_printoptions(suppress=True, precision=4)
    # print(spectral)
    # print(true_spectral)

    assert_array_almost_equal(spectral, true_spectral)

    st.to_physical(
        spectral,
        physical,
        basis_functions=[BasisFunctions.COSINE, BasisFunctions.COSINE],
    )
    assert_array_almost_equal(physical, true_physical)


def test_transform_sine_both(arrays, st, parameters):
    spectral, physical = arrays

    p = parameters
    x = np.linspace(0, 1.0, p.nx, endpoint=True)
    z = np.linspace(0, 1.0, p.nz, endpoint=True)
    X, Z = np.meshgrid(x, z, indexing="ij")

    true_physical = np.sin(np.pi * Z) * np.sin(np.pi * X) + 2.0 * np.sin(
        2 * np.pi * Z
    ) * np.sin(3 * np.pi * X)

    st.to_spectral(
        true_physical,
        spectral,
        basis_functions=[BasisFunctions.SINE, BasisFunctions.SINE],
    )

    true_spectral = np.zeros_like(spectral)
    true_spectral[1, 1] = 1.0
    true_spectral[-1, 1] = -1.0

    true_spectral[3, 2] = 2.0
    true_spectral[-3, 2] = -2.0

    assert_array_almost_equal(spectral, true_spectral)

    st.to_physical(
        spectral,
        physical,
        basis_functions=[BasisFunctions.SINE, BasisFunctions.SINE],
    )
    assert_array_almost_equal(physical, true_physical)


def test_transform_sine_x_periodic_z(arrays, st, parameters):
    spectral, physical = arrays

    p = parameters
    x = np.linspace(0, 1.0, p.nx, endpoint=True)
    z = np.linspace(0, 1.0, p.nz, endpoint=False)
    X, Z = np.meshgrid(x, z, indexing="ij")

    true_physical = np.sin(np.pi * X) + 2.0 * np.sin(2 * np.pi * X)

    st.to_spectral(
        true_physical,
        spectral,
        basis_functions=[BasisFunctions.SINE, BasisFunctions.COMPLEX_EXP],
    )

    true_spectral = np.zeros_like(spectral)
    true_spectral[1, 0] = 1.0
    true_spectral[-1, 0] = -1.0  # x is sine, complex conj is negative

    true_spectral[2, 0] = 2.0
    true_spectral[-2, 0] = -2.0

    assert_array_almost_equal(spectral, true_spectral)

    st.to_physical(
        spectral,
        physical,
        basis_functions=[BasisFunctions.SINE, BasisFunctions.COMPLEX_EXP],
    )
    assert_array_almost_equal(physical, true_physical)


def test_transform_sine_x_cosine_z(arrays, st, parameters):
    spectral, physical = arrays

    p = parameters
    x = np.linspace(0, 1.0, p.nx, endpoint=True)
    z = np.linspace(0, 1.0, p.nz, endpoint=True)
    X, Z = np.meshgrid(x, z, indexing="ij")

    true_physical = np.sin(np.pi * X) + 2.0 * np.sin(2 * np.pi * X)

    st.to_spectral(
        true_physical,
        spectral,
        basis_functions=[BasisFunctions.SINE, BasisFunctions.COSINE],
    )

    true_spectral = np.zeros_like(spectral)
    true_spectral[1, 0] = 1.0
    true_spectral[-1, 0] = -1.0  # x is sine, complex conj is negative

    true_spectral[2, 0] = 2.0
    true_spectral[-2, 0] = -2.0

    assert_array_almost_equal(spectral, true_spectral)

    st.to_physical(
        spectral,
        physical,
        basis_functions=[BasisFunctions.SINE, BasisFunctions.COSINE],
    )

    assert_array_almost_equal(physical, true_physical)


def test_transform_cosine_x_sine_z(arrays, st, parameters):
    spectral, physical = arrays

    p = parameters

    x = np.linspace(0, 1.0, p.nx, endpoint=True)
    z = np.linspace(0, 1.0, p.nz, endpoint=True)
    X, Z = np.meshgrid(x, z, indexing="ij")

    true_physical = (
        np.cos(np.pi * X) * np.sin(np.pi * Z)
        + 2.0 * np.cos(3 * np.pi * X) * np.sin(2 * np.pi * Z)
        + 3.0 * np.sin(2 * np.pi * Z)
    )

    st.to_spectral(
        true_physical,
        spectral,
        basis_functions=[BasisFunctions.COSINE, BasisFunctions.SINE],
    )

    true_spectral = np.zeros_like(spectral)
    true_spectral[1, 1] = 1.0
    true_spectral[-1, 1] = 1.0  # x is cosine, complex conjugate is the same

    true_spectral[0, 2] = 3.0

    true_spectral[3, 2] = 2.0
    true_spectral[-3, 2] = 2.0

    assert_array_almost_equal(spectral, true_spectral)

    st.to_physical(
        spectral,
        physical,
        basis_functions=[BasisFunctions.COSINE, BasisFunctions.SINE],
    )

    assert_array_almost_equal(physical, true_physical)


def test_transform_periodic_x_fdm_z(fdm_parameters):
    p = fdm_parameters
    array_factory = ArrayFactory(fdm_parameters, np)
    st = SpectralTransformer(fdm_parameters, np, array_factory=array_factory)

    spectral = array_factory.make_spectral()
    physical = array_factory.make_physical()

    x = np.linspace(0, 1.0, p.nx, endpoint=False)
    z = np.linspace(0, 1.0, p.nz)
    X, Z = np.meshgrid(x, z, indexing="ij")

    true_physical = (
        3.0 + np.cos(2 * np.pi * X) + 2.0 * np.cos(2 * 2 * np.pi * X)
    )

    st.to_spectral(
        true_physical,
        spectral,
        basis_functions=[BasisFunctions.COMPLEX_EXP, BasisFunctions.FDM],
    )

    true_spectral = np.zeros_like(spectral)
    true_spectral[0, :] = 3.0
    true_spectral[1, :] = 1.0 / 2
    true_spectral[2, :] = 2.0 / 2

    assert_array_almost_equal(spectral, true_spectral)

    st.to_physical(
        spectral,
        physical,
        basis_functions=[BasisFunctions.COMPLEX_EXP, BasisFunctions.FDM],
    )

    assert_array_almost_equal(physical, true_physical)
