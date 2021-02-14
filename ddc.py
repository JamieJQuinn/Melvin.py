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
from SpatialDifferentiator import SpatialDifferentiator
from Integrator import Integrator
from Timer import Timer
from ScalarTracker import ScalarTracker

xp=cupy

def load_initial_conditions(params, w, tmp, xi):
    x = np.linspace(0, params.lx, params.nx, endpoint = False)
    z = np.linspace(0, params.lz, params.nz, endpoint = False)
    X, Z = np.meshgrid(x, z, indexing='ij')

    rng = default_rng(0)

    epsilon = 1e-2

    w0_p = np.zeros_like(X)
    tmp0_p = np.zeros_like(X)
    xi0_p = np.zeros_like(X)

    # tmp0_p = np.array(Z>0.5) + 0.0
    # xi0_p = np.array(Z>0.5) + 0.0

    # tmp0_p = Z
    # xi0_p = 1.0-Z

    # w0_p += epsilon*(2*rng.random((params.nx, params.nz))-1.0)
    tmp0_p += epsilon*(2*rng.random((params.nx, params.nz))-1.0)
    xi0_p += epsilon*(2*rng.random((params.nx, params.nz))-1.0)

    w.load_ics(w0_p)
    tmp.load_ics(tmp0_p)
    xi.load_ics(xi0_p)

def calc_kinetic_energy(ux, uz, xp, params):
    nx, nz = params.nx, params.nz
    ke = uz.getp()**2 + ux.getp()**2
    total_ke = 0.5*xp.sum(ke)/(nx*nz)
    return total_ke

def main():
    PARAMS = {
        "nx": 2**9,
        "nz": 2**10,
        "lx": 335.0,
        "lz": 536.0,
        "initial_dt": 1e-3,
        "Pr":7,
        "R":1.1,
        "tau":1.0/3.0,
        "final_time": 800,
        "spatial_derivative_order": 4,
        "integrator_order": 4,
        "integrator": "semi-implicit",
        "dump_cadence": 1
    }
    data_trans = DataTransferer(xp)
    params = Parameters(PARAMS)
    st = SpectralTransformer(params, xp)
    integrator = Integrator(params, xp)
    ke_tracker = ScalarTracker(params, xp, "kinetic_energy.npy")
    timer = Timer()

    # Create mode number matrix
    n = np.concatenate((np.arange(0, params.nn+1),  np.arange(-params.nn, 0)))
    m = np.arange(0, params.nm)
    n, m = data_trans.from_host(np.meshgrid(n, m, indexing='ij'))

    sd = SpatialDifferentiator(params, xp, n, m)
    lap_solver = LaplacianSolver(params, xp, n, m)

    w = Variable(params, xp, sd=sd, st=st, dt=data_trans, dump_name="w")
    dw = TimeDerivative(params, xp)
    tmp = Variable(params, xp, sd=sd, st=st, dt=data_trans, dump_name="tmp")
    dtmp = TimeDerivative(params, xp)
    xi = Variable(params, xp, sd=sd, st=st, dt=data_trans, dump_name="xi")
    dxi = TimeDerivative(params, xp)

    psi = Variable(params, xp, sd=sd, st=st)
    ux = Variable(params, xp, sd=sd, st=st)
    uz = Variable(params, xp, sd=sd, st=st)

    load_initial_conditions(params, w, tmp, xi)

    t = 0.0
    dt = params.initial_dt

    print_counter = 0.0
    ke_counter = 0
    cfl_counter = 0
    loop_counter = 0

    total_start = time.time()

    wallclock_remaining = 0.0

    while t < params.final_time:
        if print_counter <= t:
            print_counter += params.dump_cadence
            # w.plot()
            tmp.save()
            xi.save()
            ke_tracker.save()
            # w.save()
            print("{0:.2f}% complete".format(t/params.final_time *100),"t = {0:.2f}".format(t), "dt = {0:.2e}".format(dt), "Remaining: {0:.2f} hr".format(wallclock_remaining/3600))
        lap_solver.solve(w.gets(), psi.gets())

        if ke_counter < loop_counter:
            # Calculate kinetic energy
            ke_counter += params.ke_cadence
            ke_tracker.append(t, data_trans.to_host(calc_kinetic_energy(ux, uz, xp, params)))

            # Calculate remaining time in simulation
            timer.split()
            wallclock_per_timestep = timer.diff/params.ke_cadence
            wallclock_remaining = wallclock_per_timestep*(params.final_time - t)/dt

        if cfl_counter < loop_counter:
            # # Adapt timestep
            cfl_counter += params.cfl_cadence
            dt = integrator.set_dt(ux, uz)

        # Remove mean flows
        psi._sdata[0,:] = 0.0 # Horizontal flow
        w._sdata[0,:] = 0.0
        psi._sdata[:,0] = 0.0 # Vertical flows
        w._sdata[:,0] = 0.0

        # Remove mean x variation
        # tmp[:,0] = 0.0
        # xi[:,0] = 0.0

        ux[:] = -psi.sddz()
        ux.to_physical()
        uz[:] = psi.sddx()
        uz.to_physical()

        w.to_physical()
        lin_op = params.Pr*lap_solver.lap
        dw[:] = -w.vec_dot_nabla(ux.getp(), uz.getp()) + params.Pr*xi.sddx() - params.Pr*tmp.sddx()
        integrator.integrate(w, dw, lin_op)

        tmp.to_physical()
        lin_op = lap_solver.lap
        dtmp[:] = -tmp.vec_dot_nabla(ux.getp(), uz.getp()) - uz[:]
        integrator.integrate(tmp, dtmp, lin_op)

        xi.to_physical()
        lin_op = params.tau*lap_solver.lap
        dxi[:] = -xi.vec_dot_nabla(ux.getp(), uz.getp()) - uz[:]/params.R
        integrator.integrate(xi, dxi, lin_op)

        t += dt
        loop_counter += 1

    total_end = time.time() - total_start
    print(total_end)

if __name__=="__main__":
    main()
