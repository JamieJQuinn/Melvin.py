import numpy as np
import pytest
import time
from numpy.testing import assert_array_almost_equal

from pytest import approx
from Variable import Variable
from SpatialDifferentiator import SpatialDifferentiator
from ArrayFactory import ArrayFactory
from SpectralTransformer import SpectralTransformer

def test_spatial_derivatives(parameters, st, sd, array_factory):
    p = parameters
    x = np.linspace(0, p.lx, p.nx, endpoint=False)
    z = np.linspace(0, p.lz, p.nz, endpoint=False)
    X, Z = np.meshgrid(x, z, indexing='ij')

    var = Variable(parameters, np, st=st, sd=sd, array_factory=array_factory)
    var.setp(np.cos(2*np.pi*X)*np.cos(2*np.pi*Z))

    dvardx = var.pddx()

    dvardx_true = -2*np.pi*np.sin(2*np.pi*x)
    dvardx_test = dvardx[:,0]

    assert dvardx_test == approx(dvardx_true, rel=1e-3)

    dvardz = var.pddz()

    dvardz_true = -2*np.pi*np.sin(2*np.pi*z)
    dvardz_test = dvardz[0,:]

    assert dvardz_test == approx(dvardz_true, rel=1e-3)

def test_fdm_nabla2(fdm_parameters):
    p = fdm_parameters
    x = np.linspace(0, p.lx, p.nx, endpoint=False)
    z = np.linspace(0, p.lz, p.nz, endpoint=False)
    X, Z = np.meshgrid(x, z, indexing='ij')

    array_factory = ArrayFactory(p, np)
    spatial_diff = SpatialDifferentiator(p, np, array_factory=array_factory)

    var = Variable(p, np, sd=spatial_diff, array_factory=array_factory)
    var[1] = 0.5*z**2

    nabla2 = var.snabla2()

    true_nabla2 = np.zeros_like(nabla2)
    true_nabla2[1] = -(2*np.pi)**2*var[1] + 1

    assert_array_almost_equal(nabla2[1, 1:-1], true_nabla2[1, 1:-1])

def test_spectral_nabla2(parameters):
    p = parameters
    x = np.linspace(0, p.lx, p.nx, endpoint=False)
    z = np.linspace(0, p.lz, p.nz, endpoint=False)
    X, Z = np.meshgrid(x, z, indexing='ij')

    array_factory = ArrayFactory(p, np)
    spatial_diff = SpatialDifferentiator(p, np, array_factory=array_factory)

    var = Variable(p, np, sd=spatial_diff, array_factory=array_factory)
    var[:] = 1

    nabla2 = var.snabla2()

    n, m = array_factory.make_mode_number_matrices()
    true_nabla2 = -(2*np.pi/p.lx*n)**2 - (2*np.pi/p.lz*m)**2

    assert_array_almost_equal(nabla2, true_nabla2)
