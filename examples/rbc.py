#!/usr/bin/env python3

import numpy as np
from numpy.random import default_rng

import cupy
import time
import matplotlib.pyplot as plt

from melvin import Parameters, SpectralTransformer, DataTransferer, Variable,\
        TimeDerivative, SpatialDifferentiator, Integrator, Timer, ScalarTracker, \
        RunningState, ArrayFactory, LaplacianSolver, BasisFunctions

xp=cupy

def load_initial_conditions(params, w, tmp):
    x = np.linspace(0, params.lx, params.nx, endpoint = False)
    z = np.linspace(0, params.lz, params.nz, endpoint = True)
    X, Z = np.meshgrid(x, z, indexing='ij')

    rng = default_rng(0)

    epsilon = 1e-1

    w0_p = np.zeros_like(X)
    tmp0_p = np.zeros_like(X) + 0.5
    # tmp0_p = 1-Z

    w0_p += epsilon*(2*rng.random((params.nx, params.nz))-1.0)
    tmp0_p += epsilon*(2*rng.random((params.nx, params.nz))-1.0)
    # tmp0_p += epsilon*np.sin(np.pi*Z)*np.cos(2*np.pi*X)

    w.load(w0_p, is_physical=True)
    tmp.load(tmp0_p, is_physical=True)

def calc_kinetic_energy(ux, uz, xp, params):
    nx, nz = params.nx, params.nz
    ke = uz.getp()**2 + ux.getp()**2
    total_ke = 0.5*xp.sum(ke)/(nx*nz)
    return total_ke

def form_dumpname(index):
    return f'dump{index:04d}.npz'

def dump(index, xp, data_trans, w, dw, tmp, dtmp):
    fname = form_dumpname(index)
    np.savez(fname, 
                w =data_trans.to_host( w[:]),
                dw=data_trans.to_host(dw.get_all()),
                tmp =data_trans.to_host( tmp[:]),
                dtmp=data_trans.to_host(dtmp.get_all()), 
                curr_idx = dw.get_curr_idx())

def load(index, xp, w, dw, tmp, dtmp):
    fname = form_dumpname(index)
    dump_arrays = xp.load(fname)
    w.load(dump_arrays['w'])
    dw.load(dump_arrays['dw'])
    tmp.load(dump_arrays['tmp'])
    dtmp.load(dump_arrays['dtmp'])

    # This assumes all variables are integrated together
    dw.set_curr_idx(dump_arrays['curr_idx'])
    dtmp.set_curr_idx(dump_arrays['curr_idx'])

def main():
    PARAMS = {
        "nx": 2**6,
        "nz": 2**6,
        "lx": 2.0,
        "lz": 1.0,
        "initial_dt": 1e-5,
        "cfl_cutoff": 0.5,
        "Pr":1.0,
        "Ra":1e6,
        "final_time": 1e-1,
        "spatial_derivative_order": 2,
        "integrator_order": 2,
        "integrator": "explicit",
        "save_cadence": 5e-5,
        # "load_from": 49,
        "dump_cadence": 1e-1,
        "discretisation": ['spectral', 'fdm']
    }
    params = Parameters(PARAMS)
    state = RunningState(params)

    data_trans = DataTransferer(xp)
    array_factory = ArrayFactory(params, xp)

    # Algorithms
    sd = SpatialDifferentiator(params, xp, array_factory=array_factory)
    st = SpectralTransformer(params, xp, array_factory)
    integrator = Integrator(params, xp)

    # Trackers
    ke_tracker = ScalarTracker(params, xp, "kinetic_energy.npz")

    # Simulation variables

    bs = [BasisFunctions.COMPLEX_EXP, BasisFunctions.FDM]

    w = Variable(params, xp, sd=sd, st=st, dt=data_trans, array_factory=array_factory, dump_name="w", basis_functions=bs)
    dw = TimeDerivative(params, xp)
    tmp = Variable(params, xp, sd=sd, st=st, dt=data_trans, array_factory=array_factory, dump_name="tmp", basis_functions=bs)
    dtmp = TimeDerivative(params, xp)

    psi = Variable(params, xp, sd=sd, st=st, array_factory=array_factory, dump_name='psi', basis_functions=bs)
    ux = Variable(params, xp, sd=sd, st=st, array_factory=array_factory, dump_name='ux', basis_functions=bs)
    uz = Variable(params, xp, sd=sd, st=st, array_factory=array_factory, dump_name='uz', basis_functions=bs)

    # Laplacian solver depends on psi for its basis calculation
    laplacian_solver = LaplacianSolver(params, xp, psi._basis_functions, spatial_diff=sd, array_factory=array_factory)

    # Load initial conditions

    if params.load_from is not None:
        load(params.load_from, xp, w, dw, tmp, dtmp)
        integrator.override_dt(state.dt)
    else:
        load_initial_conditions(params, w, tmp)

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
            w.save()
            # ke_tracker.save()
            # nusselt_tracker.save()

        if state.dump_counter <= state.t:
            state.dump_counter += params.dump_cadence
            dump(state.dump_index, xp, data_trans,
                 w, dw, tmp, dtmp)
            state.save(state.dump_index)
            state.dump_index += 1

        if state.ke_counter < state.loop_counter:
            # Calculate kinetic energy
            state.ke_counter += params.ke_cadence
            ke_tracker.append(state.t, calc_kinetic_energy(ux, uz, xp, params))

            # Calculate remaining time in simulation
            timer.split()
            wallclock_per_timestep = timer.diff/params.ke_cadence
            wallclock_remaining = wallclock_per_timestep*(params.final_time - state.t)/state.dt

        if state.cfl_counter < state.loop_counter:
            # Adapt timestep
            state.cfl_counter += params.cfl_cadence
            state.dt = integrator.set_dt(ux, uz)
            if state.dt is None:
                return

        # omega boundary conditions
        w[1:,0] = 0.0
        w[1:,-1] = 0.0

        # SOLVER STARTS HERE
        laplacian_solver.solve(-w.gets(), out=psi._sdata)

        psi[1:,0] = 0.0
        psi[1:,-1] = 0.0

        psi[0,:] = 0.0
        w[0,:] = 0.0

        tmp[0,0] = 1.0
        tmp[0,-1] = 0.0

        tmp[1:,0] = 0.0
        tmp[1:,-1] = 0.0

        psi.to_physical()
        ux.setp(-psi.pddz())

        uz[:] = psi.sddx()
        uz.to_physical()

        diffusion_term = params.Pr*w.snabla2()
        dw[:] = -w.vec_dot_nabla(ux.getp(), uz.getp()) - params.Pr*params.Ra*tmp.sddx()
        integrator.integrate(w, dw, diffusion_term)

        diffusion_term = tmp.snabla2()
        dtmp[:] = -tmp.vec_dot_nabla(ux.getp(), uz.getp())
        integrator.integrate(tmp, dtmp, diffusion_term)

        tmp.to_physical()

        state.t += state.dt
        state.loop_counter += 1

    total_end = time.time() - total_start
    print(f"Total time: {total_end/3600:.2f} hr")
    print(f"Total time: {total_end:.2f} s")

if __name__=="__main__":
    main()
