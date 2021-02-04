#!/usr/bin/env python3

import numpy as np
from numpy.random import default_rng

import cupy
import time
from utility import sech
import matplotlib.pyplot as plt

from Parameters import Parameters
from SpectralTransformer import SpectralTransformer
from DataTransferer import DataTransferer
from Variable import Variable
from TimeDerivative import TimeDerivative
from LaplacianSolver import LaplacianSolver
from Integrator import Integrator

MODULE=cupy

def load_initial_conditions(params, w):
    x = np.linspace(0, params.lx, params.nx, endpoint = False)
    z = np.linspace(0, params.lz, params.nz, endpoint = False)
    X, Z = np.meshgrid(x, z, indexing='ij')

    rng = default_rng(0)

    epsilon = 0.01
    sigma = 0.2
    h = 0.05
    pert_n = 1

    ## Set vorticity
    w0_p = np.power(sech((Z-0.5)/h), 2)/h
    w0_pert_p = epsilon * params.kn*np.cos(pert_n*params.kn*X+np.pi)\
    *(np.exp(-((Z-0.5)/sigma)**2))

    w0_p += -np.power(sech((Z-1.5)/h), 2)/h
    w0_pert_p += epsilon * params.kn*-np.cos(pert_n*params.kn*X+np.pi)\
    *(np.exp(-((Z-1.5)/sigma)**2))

    # w0_pert_p = epsilon*(2*rng.random((params.nx, params.nz))-1.0)

    w0_p += w0_pert_p

    w.load_ics(w0_p)

    ## Set ink
    # c0_p = np.sin(1*params.km*Z)

    # c.set_physical(c0_p)
    # c.to_spectral()

def main():
    PARAMS = {
        "nx": 4**5,
        "nz": 2**11,
        "lx": 1.0,
        "lz": 2.0,
        "dt": 0.2/4**5,
        "Re": 1e5,
        "integrator_order": 2,
        "max_time": 10.0,
        "dump_cadence": 0.1
    }
    PARAMS['dt'] = 0.2*PARAMS['lx']/PARAMS['nx']
    dt = DataTransferer(MODULE)
    params = Parameters(PARAMS)
    st = SpectralTransformer(params, MODULE)
    lap_solver = LaplacianSolver(params, MODULE)
    integrator = Integrator(params)

    # Create mode number matrix
    n = np.concatenate((np.arange(0, params.nn+1),  np.arange(-params.nn, 0)))
    m = np.arange(0, params.nm)
    n, m = dt.from_host(np.meshgrid(n, m, indexing='ij'))

    w = Variable(params, MODULE, st, dt, dump_name="w")
    dw = TimeDerivative(params, MODULE)

    psi = Variable(params, MODULE, st)
    ux = Variable(params, MODULE, st)
    uz = Variable(params, MODULE, st)

    load_initial_conditions(params, w)

    t = 0.0
    print_track = 0.0

    total_ke = 0.0
    max_vel = 0.0

    print("dt=", params.dt)

    start = time.time()
    while t < params.max_time:
        if print_track <= t:
            print_track += params.dump_cadence
            # w.plot()
            w.save()
            print(t, t/params.max_time *100, total_ke)
            print(params.dz/params.dt, max_vel)
        lap_solver.solve(w.gets(), psi.gets())
        psi._sdata[0,0] = 0.0

        ux[:] = 1j*params.km*m*psi[:]
        ux.to_physical()
        uz[:] = -1j*params.kn*n*psi[:]
        uz.to_physical()

        ke = uz.getp()**2 + ux.getp()**2
        max_vel = MODULE.max(MODULE.sqrt(ke))
        total_ke = 0.5*MODULE.sum(ke)/(params.nx*params.nz)

        w.to_physical()
        dw[:] = -w.vec_dot_nabla(ux.getp(), uz.getp())
        RHS = (1+(1-params.alpha)*params.dt/params.Re*lap_solver.lap)*w[:] + integrator.predictor(dw)
        w[:] = RHS/(1-params.alpha*params.dt/params.Re*lap_solver.lap)
        dw.advance()

        # Predictor
        # w.to_physical()
        # dw[:] = -w.vec_dot_nabla(ux.getp(), uz.getp()) + 1.0/params.Re*lap_solver.lap*w[:]
        # w[:] += integrator.predictor(dw)
        # dw.advance()

        ## Corrector
        # w.to_physical()
        # dw[:] = -w.vec_dot_nabla(ux.getp(), uz.getp()) + 1.0/params.Re*lap_solver.lap*w[:]
        # w[:] += integrator.corrector(dw)
        # dw.advance()

        t += params.dt

    end = time.time() - start
    print(end)

if __name__=="__main__":
    main()
