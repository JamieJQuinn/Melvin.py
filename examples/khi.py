#!/usr/bin/env python3

import numpy as np
from numpy.random import default_rng

import cupy
import time
import matplotlib.pyplot as plt
from melvin.utility import sech

from melvin import (
    Parameters,
    SpectralTransformer,
    DataTransferer,
    Variable,
    TimeDerivative,
    LaplacianSolver,
    Integrator,
)

MODULE = cupy


def load_initial_conditions(params, w, ink):
    x = np.linspace(0, params.lx, params.nx, endpoint=False)
    z = np.linspace(0, params.lz, params.nz, endpoint=False)
    X, Z = np.meshgrid(x, z, indexing="ij")

    R = np.sqrt((X - (params.lx / 2)) ** 2 + (Z - 0.5) ** 2)

    rng = default_rng(0)

    epsilon = 0.1
    sigma = 0.2
    h = 0.05
    pert_n = 1

    ## Set vorticity
    w0_p = np.power(sech((R - 0.25) / h), 2) / h
    # w0_pert_p = epsilon * params.kn*np.cos(pert_n*params.kn*X+np.pi)\
    # *(np.exp(-((Z-0.5)/sigma)**2))

    # w0_p += -np.power(sech((Z-1.5)/h), 2)/h
    # w0_pert_p += epsilon * params.kn*-np.cos(pert_n*params.kn*X+np.pi)\
    # *(np.exp(-((Z-1.5)/sigma)**2))

    w0_pert_p = epsilon * (2 * rng.random((params.nx, params.nz)) - 1.0)

    w0_p += w0_pert_p

    w.load_ics(w0_p)

    ## Set ink
    # ink0_p = np.array(np.logical_and(Z>0.5, Z<1.5)) + 1
    ink0_p = np.array(Z > 0.5) + 1
    ink.load_ics(ink0_p)


def main():
    PARAMS = {
        "nx": 4 ** 6,
        "nz": 2 ** 11,
        "lx": 16.0 / 9.0,
        "lz": 1.0,
        "dt": 0.2 / 4 ** 5,
        "Re": 1e5,
        "integrator_order": 2,
        "max_time": 3.0,
        "dump_cadence": 0.01,
    }
    PARAMS["dt"] = 0.1 * PARAMS["lx"] / PARAMS["nx"]
    dt = DataTransferer(MODULE)
    params = Parameters(PARAMS)
    st = SpectralTransformer(params, MODULE)
    lap_solver = LaplacianSolver(params, MODULE)
    integrator = Integrator(params)

    # Create mode number matrix
    n = np.concatenate((np.arange(0, params.nn + 1), np.arange(-params.nn, 0)))
    m = np.arange(0, params.nm)
    n, m = dt.from_host(np.meshgrid(n, m, indexing="ij"))

    w = Variable(params, MODULE, st, dt, dump_name="w")
    dw = TimeDerivative(params, MODULE)

    ink = Variable(params, MODULE, st, dt, dump_name="ink")
    dink = TimeDerivative(params, MODULE)

    psi = Variable(params, MODULE, st)
    ux = Variable(params, MODULE, st)
    uz = Variable(params, MODULE, st)

    load_initial_conditions(params, w, ink)

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
            print(t, t / params.max_time * 100, total_ke)
            print(params.dz / params.dt, max_vel)
        lap_solver.solve(w.gets(), psi.gets())
        psi._sdata[0, 0] = 0.0

        ux[:] = 1j * params.km * m * psi[:]
        ux.to_physical()
        uz[:] = -1j * params.kn * n * psi[:]
        uz.to_physical()

        ke = uz.getp() ** 2 + ux.getp() ** 2
        max_vel = MODULE.max(MODULE.sqrt(ke))
        total_ke = 0.5 * MODULE.sum(ke) / (params.nx * params.nz)

        w.to_physical()
        dw[:] = -w.vec_dot_nabla(ux.getp(), uz.getp())
        RHS = (1 + (1 - params.alpha) * params.dt / params.Re * lap_solver.lap) * w[
            :
        ] + integrator.predictor(dw)
        w[:] = RHS / (1 - params.alpha * params.dt / params.Re * lap_solver.lap)
        dw.advance()

        # ink
        # ink.to_physical()
        # dink[:] = -ink.vec_dot_nabla(ux.getp(), uz.getp()) + 1.0/params.Re*lap_solver.lap*ink[:]
        # ink[:] += integrator.predictor(dink)
        # dink.advance()

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


if __name__ == "__main__":
    main()
