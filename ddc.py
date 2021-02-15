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

def form_dumpname(index):
    return f'dump{index:04d}.npz'

def dump(index, xp, data_trans, w, dw, tmp, dtmp, xi, dxi):
    fname = form_dumpname(index)
    np.savez(fname, 
                w =data_trans.to_host( w[:]),
                dw=data_trans.to_host(dw[:]),
                tmp =data_trans.to_host( tmp[:]),
                dtmp=data_trans.to_host(dtmp[:]), 
                xi =data_trans.to_host( xi[:]),
                dxi=data_trans.to_host(dxi[:]),
                curr_idx = dw.get_curr_idx())

def load(index, xp, w, dw, tmp, dtmp, xi, dxi):
    fname = form_dumpname(index)
    dump_arrays = xp.load(fname)
    w[:] = dump_arrays['w']
    dw[:] = dump_arrays['dw']
    tmp[:] = dump_arrays['tmp']
    dtmp[:] = dump_arrays['dtmp']
    xi[:] = dump_arrays['xi']
    dxi[:] = dump_arrays['dxi']

    # This assumes all variables are integrated together
    dw.set_curr_idx(dump_arrays['curr_idx'])
    dtmp.set_curr_idx(dump_arrays['curr_idx'])
    dxi.set_curr_idx(dump_arrays['curr_idx'])

def main():
    PARAMS = {
        "nx": 2**9,
        "nz": 2**9,
        "lx": 335.0,
        "lz": 536.0,
        "initial_dt": 1e-3,
        "Pr":7.0,
        "R":1.1,
        "tau":1.0/3.0,
        "final_time": 11,
        "spatial_derivative_order": 4,
        "integrator_order": 4,
        "integrator": "semi-implicit",
        "load_from": 1,
        "save_cadence": 10,
        "dump_cadence": 5
    }
    data_trans = DataTransferer(xp)
    params = Parameters(PARAMS)
    st = SpectralTransformer(params, xp)
    integrator = Integrator(params, xp)
    ke_tracker = ScalarTracker(params, xp, "kinetic_energy.npy")
    state = RunningState(params)
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

    if params.load_from is not None:
        load(params.load_from, xp, w, dw, tmp, dtmp, xi, dxi)
    else:
        load_initial_conditions(params, w, tmp, xi)

    total_start = time.time()
    wallclock_remaining = 0.0

    while state.t < params.final_time:
        if state.save_counter <= state.t:
            state.save_counter += params.save_cadence
            print(f"{state.t/params.final_time *100:.2f}% complete",
                  f"t = {state.t:.2f}", 
                  f"dt = {state.dt:.2e}", 
                  f"Remaining: {wallclock_remaining/3600:.2f} hr")
            # w.plot()
            tmp.save()
            xi.save()
            ke_tracker.save()
            # w.save()

        if state.dump_counter <= state.t:
            state.dump_counter += params.dump_cadence
            dump(state.dump_index, xp, data_trans,
                 w, dw, tmp, dtmp, xi, dxi)
            state.save(state.dump_index)
            state.dump_index += 1

        if state.ke_counter < state.loop_counter:
            # Calculate kinetic energy
            state.ke_counter += params.ke_cadence
            ke_tracker.append(state.t, data_trans.to_host(calc_kinetic_energy(ux, uz, xp, params)))

            # Calculate remaining time in simulation
            timer.split()
            wallclock_per_timestep = timer.diff/params.ke_cadence
            wallclock_remaining = wallclock_per_timestep*(params.final_time - state.t)/state.dt

        if state.cfl_counter < state.loop_counter:
            # Adapt timestep
            state.cfl_counter += params.cfl_cadence
            state.dt = integrator.set_dt(ux, uz)

        # SOLVER STARTS HERE

        lap_solver.solve(w.gets(), psi.gets())

        # Remove mean flows
        psi._sdata[0,:] = 0.0 # Horizontal flow
        w._sdata[0,:] = 0.0
        psi._sdata[:,0] = 0.0 # Vertical flows
        w._sdata[:,0] = 0.0

        # Remove mean x variation
        tmp[:,0] = 0.0
        xi[:,0] = 0.0

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

        state.t += state.dt
        state.loop_counter += 1

    total_end = time.time() - total_start
    print(f"Total time: {total_end/3600:.2f} hr")

if __name__=="__main__":
    main()
