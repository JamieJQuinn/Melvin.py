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

MODULE=cupy

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
        "dump_cadence": 10
    }
    data_trans = DataTransferer(MODULE)
    params = Parameters(PARAMS)
    st = SpectralTransformer(params, MODULE)
    integrator = Integrator(params)

    # Create mode number matrix
    n = np.concatenate((np.arange(0, params.nn+1),  np.arange(-params.nn, 0)))
    m = np.arange(0, params.nm)
    n, m = data_trans.from_host(np.meshgrid(n, m, indexing='ij'))

    sd = SpatialDifferentiator(params, MODULE, n, m)
    lap_solver = LaplacianSolver(params, MODULE, n, m)

    w = Variable(params, MODULE, sd=sd, st=st, dt=data_trans, dump_name="w")
    dw = TimeDerivative(params, MODULE)
    tmp = Variable(params, MODULE, sd=sd, st=st, dt=data_trans, dump_name="tmp")
    dtmp = TimeDerivative(params, MODULE)
    xi = Variable(params, MODULE, sd=sd, st=st, dt=data_trans, dump_name="xi")
    dxi = TimeDerivative(params, MODULE)

    psi = Variable(params, MODULE, sd=sd, st=st)
    ux = Variable(params, MODULE, sd=sd, st=st)
    uz = Variable(params, MODULE, sd=sd, st=st)

    load_initial_conditions(params, w, tmp, xi)

    t = 0.0
    dt = params.initial_dt

    print_tracker = 0.0

    ke_cadence = 10
    ke_tracker = 0

    loop_counter = 0

    total_ke = 0.0

    start = time.time()
    while t < params.final_time:
        if print_tracker <= t:
            print_tracker += params.dump_cadence
            # w.plot()
            tmp.save()
            xi.save()
            w.save()
            print("{0:.2f}% complete".format(t/params.final_time *100),"t = {0:.2f}".format(t), "KE = {0:.2f}".format(data_trans.to_host(total_ke)), "dt = {0:.2e}".format(dt))
        lap_solver.solve(w.gets(), psi.gets())

        # Remove mean flows
        psi._sdata[0,:] = 0.0
        w._sdata[0,:] = 0.0
        psi._sdata[:,0] = 0.0
        w._sdata[:,0] = 0.0
        
        # Remove averages
        # tmp._sdata[0,:] = 0.0
        # xi._sdata[0,:] = 0.0
        # tmp._sdata[:,0] = 0.0
        # xi._sdata[:,0] = 0.0

        ux[:] = psi.sddz()
        ux.to_physical()
        uz[:] = -psi.sddx()
        uz.to_physical()

        if ke_tracker < loop_counter:
            ke_tracker += ke_cadence
            ke = uz.getp()**2 + ux.getp()**2
            cfl_dt = min(params.dx/MODULE.max(ux.getp()), params.dz/MODULE.max(uz.getp()))
            if dt > cfl_dt or np.isnan(cfl_dt):
                print("CFL condition breached")
                return
            while dt > 0.2*cfl_dt:
                # new_dt = dt*0.9
                dt = dt*0.9
                print("new dt:", dt)
            # if dt < 0.1*cfl_dt:
                # # new_dt = dt*1.1
                # dt = dt*1.1
            integrator.set_dt(dt)

            total_ke = 0.5*MODULE.sum(ke)/(params.nx*params.nz)

        w.to_physical()
        lin_op = dt*params.Pr*lap_solver.lap
        dw[:] = -w.vec_dot_nabla(ux.getp(), uz.getp()) + params.Pr*xi.sddx() - params.Pr*tmp.sddx()

        RHS = (1+(1-params.alpha)*lin_op)*w[:] + integrator.predictor(dw)
        w[:] = RHS/(1-params.alpha*lin_op)
        dw.advance()

        tmp.to_physical()
        lin_op = dt*lap_solver.lap
        dtmp[:] = -tmp.vec_dot_nabla(ux.getp(), uz.getp()) + uz[:]

        RHS = (1+(1-params.alpha)*lin_op)*tmp[:] + integrator.predictor(dtmp)
        tmp[:] = RHS/(1-params.alpha*lin_op)
        dtmp.advance()

        xi.to_physical()
        lin_op = dt*params.tau*lap_solver.lap
        dxi[:] = -xi.vec_dot_nabla(ux.getp(), uz.getp()) + uz[:]/params.R

        RHS = (1+(1-params.alpha)*lin_op)*xi[:] + integrator.predictor(dxi)
        xi[:] = RHS/(1-params.alpha*lin_op)
        dxi.advance()

        # Predictor
        # w.to_physical()
        # dw[:] = -w.vec_dot_nabla(ux.getp(), uz.getp()) + PARAMS['Pr']*xi.sddx() - PARAMS['Pr']*tmp.sddx() + PARAMS['Pr']*lap_solver.lap*w[:]
        # w[:] += integrator.predictor(dw)
        # dw.advance()

        # tmp.to_physical()
        # dtmp[:] = -tmp.vec_dot_nabla(ux.getp(), uz.getp()) - uz[:] + lap_solver.lap*tmp[:]
        # tmp[:] += integrator.predictor(dtmp)
        # dtmp.advance()

        # xi.to_physical()
        # dxi[:] = -xi.vec_dot_nabla(ux.getp(), uz.getp()) - uz[:]/PARAMS['R'] + PARAMS['tau']*lap_solver.lap*xi[:]
        # xi[:] += integrator.predictor(dxi)
        # dxi.advance()

        # w.to_physical()
        # dw[:] = -w.vec_dot_nabla(ux.getp(), uz.getp()) + PARAMS['Pr']*xi.sddx() - PARAMS['Pr']*tmp.sddx() + PARAMS['Pr']*lap_solver.lap*w[:]
        # w[:] += integrator.corrector(dw)

        # tmp.to_physical()
        # dtmp[:] = -tmp.vec_dot_nabla(ux.getp(), uz.getp()) - uz[:] + lap_solver.lap*tmp[:]
        # tmp[:] += integrator.corrector(dtmp)

        # xi.to_physical()
        # dxi[:] = -xi.vec_dot_nabla(ux.getp(), uz.getp()) - uz[:]/PARAMS['R'] + PARAMS['tau']*lap_solver.lap*xi[:]
        # xi[:] += integrator.corrector(dxi)

        ## Corrector
        # w.to_physical()
        # dw[:] = -w.vec_dot_nabla(ux.getp(), uz.getp()) - PARAMS['Ra']*PARAMS['Pr']*params.kn*tmp[:] + PARAMS['Pr']*lap_solver.lap*w[:]
        # w[:] += integrator.corrector(dw)
        # dw.advance()

        # tmp.to_physical()
        # dtmp[:] = -tmp.vec_dot_nabla(ux.getp(), uz.getp()) + lap_solver.lap*tmp[:]
        # tmp[:] += integrator.corrector(dtmp)
        # dtmp.advance()

        t += dt
        loop_counter += 1

    end = time.time() - start
    print(end)

if __name__=="__main__":
    main()
