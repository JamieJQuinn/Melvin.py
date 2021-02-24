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
from RunningState import RunningState

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

    tmp0_p += epsilon*(2*rng.random((params.nx, params.nz))-1.0)
    xi0_p += epsilon*(2*rng.random((params.nx, params.nz))-1.0)

    w.load(w0_p, is_physical=True)
    tmp.load(tmp0_p, is_physical=True)
    xi.load(xi0_p, is_physical=True)

def calc_kinetic_energy(ux, uz, xp, params):
    nx, nz = params.nx, params.nz
    ke = uz.getp()**2 + ux.getp()**2
    total_ke = 0.5*xp.sum(ke)/(nx*nz)
    return total_ke

def calc_nusselt_number(tmp, uz, xp, params):
    # From Stellmach et al 2011 (DOI: 10.1017/jfm.2011.99)
    flux = xp.mean(tmp.getp()*uz.getp())
    return 1.0 - flux

def form_dumpname(index):
    return f'dump{index:04d}.npz'

def dump(index, xp, data_trans, w, dw, tmp, dtmp, xi, dxi):
    fname = form_dumpname(index)
    np.savez(fname, 
                w =data_trans.to_host( w[:]),
                dw=data_trans.to_host(dw.get_all()),
                tmp =data_trans.to_host( tmp[:]),
                dtmp=data_trans.to_host(dtmp.get_all()), 
                xi =data_trans.to_host( xi[:]),
                dxi=data_trans.to_host(dxi.get_all()),
                curr_idx = dw.get_curr_idx())

def load(index, xp, w, dw, tmp, dtmp, xi, dxi):
    fname = form_dumpname(index)
    dump_arrays = xp.load(fname)
    w.load(dump_arrays['w'])
    dw.load(dump_arrays['dw'])
    tmp.load(dump_arrays['tmp'])
    dtmp.load(dump_arrays['dtmp'])
    xi.load(dump_arrays['xi'])
    dxi.load(dump_arrays['dxi'])

    # This assumes all variables are integrated together
    dw.set_curr_idx(dump_arrays['curr_idx'])
    dtmp.set_curr_idx(dump_arrays['curr_idx'])
    dxi.set_curr_idx(dump_arrays['curr_idx'])

def main():
    PARAMS = {
        "nx": 2**9,
        "nz": 2**10,
        "lx": 335.0,
        "lz": 536.0,
        # "lx": 134.0,
        # "lz": 268.0,
        "initial_dt": 1e-3,
        "cfl_cutoff": 0.5,
        "Pr":7.0,
        "R0":1.1,
        "tau":1.0/3.0,
        "final_time": 800,
        "spatial_derivative_order": 2,
        "integrator_order": 2,
        "integrator": "semi-implicit",
        "save_cadence": 0.1,
        # "load_from": 17,
        "dump_cadence": 10
    }
    params = Parameters(PARAMS)
    state = RunningState(params)

    data_trans = DataTransferer(xp)

    # Create mode number matrix
    n = np.concatenate((np.arange(0, params.nn+1),  np.arange(-params.nn, 0)))
    m = np.arange(0, params.nm)
    n, m = data_trans.from_host(np.meshgrid(n, m, indexing='ij'))

    # Algorithms
    sd = SpatialDifferentiator(params, xp, n, m)
    lap_solver = LaplacianSolver(params, xp, n, m)
    st = SpectralTransformer(params, xp)
    integrator = Integrator(params, xp)

    # Trackers
    ke_tracker = ScalarTracker(params, xp, "kinetic_energy.npz")
    nusselt_tracker = ScalarTracker(params, xp, "nusselt.npz")

    # Simulation variables

    w = Variable(params, xp, sd=sd, st=st, dt=data_trans, dump_name="w")
    dw = TimeDerivative(params, xp)
    tmp = Variable(params, xp, sd=sd, st=st, dt=data_trans, dump_name="tmp")
    dtmp = TimeDerivative(params, xp)
    xi = Variable(params, xp, sd=sd, st=st, dt=data_trans, dump_name="xi")
    dxi = TimeDerivative(params, xp)

    psi = Variable(params, xp, sd=sd, st=st, dump_name='psi')
    ux = Variable(params, xp, sd=sd, st=st, dump_name='ux')
    uz = Variable(params, xp, sd=sd, st=st, dump_name='uz')

    # Load initial conditions

    if params.load_from is not None:
        load(params.load_from, xp, w, dw, tmp, dtmp, xi, dxi)
        integrator.override_dt(state.dt)
    else:
        load_initial_conditions(params, w, tmp, xi)

    total_start = time.time()
    wallclock_remaining = 0.0
    timer = Timer()

    # Main loop

    while state.t < params.final_time:
        if state.save_counter <= state.t:
            state.save_counter += params.save_cadence
            print(f"{state.t/params.final_time *100:.2f}% complete",
                  f"t = {state.t:.2f}", 
                  f"dt = {state.dt:.2e}", 
                  f"Remaining: {wallclock_remaining/3600:.2f} hr")
            tmp.save()
            ke_tracker.save()
            nusselt_tracker.save()

        if state.dump_counter <= state.t:
            state.dump_counter += params.dump_cadence
            dump(state.dump_index, xp, data_trans,
                 w, dw, tmp, dtmp, xi, dxi)
            state.save(state.dump_index)
            state.dump_index += 1

        if state.ke_counter < state.loop_counter:
            # Calculate kinetic energy
            state.ke_counter += params.ke_cadence
            ke_tracker.append(state.t, calc_kinetic_energy(ux, uz, xp, params))
            nusselt_tracker.append(state.t, calc_nusselt_number(tmp, uz, xp, params))

            # Calculate remaining time in simulation
            timer.split()
            wallclock_per_timestep = timer.diff/params.ke_cadence
            wallclock_remaining = wallclock_per_timestep*(params.final_time - state.t)/state.dt

        if state.cfl_counter < state.loop_counter:
            # Adapt timestep
            state.cfl_counter += params.cfl_cadence
            state.dt = integrator.set_dt(ux, uz)

        # SOLVER STARTS HERE

        lap_solver.solve(-w.gets(), psi.gets())

        # Remove mean z variation
        tmp[:,0] = 0.0
        xi[:,0] = 0.0

        ux[:] = -psi.sddz()
        ux.to_physical()
        uz[:] = psi.sddx()
        uz.to_physical()

        # lin_op = params.Pr*lap_solver.lap
        lin_op = params.Pr*lap_solver.lap
        dw[:] = -w.vec_dot_nabla(ux.getp(), uz.getp()) + params.Pr*xi.sddx() - params.Pr*tmp.sddx()
        integrator.integrate(w, dw, lin_op)

        lin_op = lap_solver.lap
        dtmp[:] = -tmp.vec_dot_nabla(ux.getp(), uz.getp()) - uz[:]
        integrator.integrate(tmp, dtmp, lin_op)

        lin_op = params.tau*lap_solver.lap
        dxi[:] = -xi.vec_dot_nabla(ux.getp(), uz.getp()) - uz[:]/params.R0
        integrator.integrate(xi, dxi, lin_op)

        state.t += state.dt
        state.loop_counter += 1

    total_end = time.time() - total_start
    print(f"Total time: {total_end/3600:.2f} hr")
    print(f"Total time: {total_end:.2f} s")

if __name__=="__main__":
    main()
