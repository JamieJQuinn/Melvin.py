#!/usr/bin/env python3

from functools import partial
import numpy as np
from numpy.random import default_rng

import cupy
import time

from melvin import Parameters, Simulation, BasisFunctions

from melvin.utility import (
    calc_kinetic_energy,
    calc_velocity_from_vorticity,
    init_var_with_noise,
    sech
)

xp = cupy


def load_initial_conditions(params, w, ink):
    x = np.linspace(0, params.lx, params.nx, endpoint=False)
    z = np.linspace(0, params.lz, params.nz, endpoint=False)
    X, Z = np.meshgrid(x, z, indexing="ij")

    R = np.sqrt((X - (params.lx / 2)) ** 2 + (Z - 0.5) ** 2)

    rng = default_rng(0)

    epsilon = 0.01
    # sigma = 0.2
    h = 0.1
    # pert_n = 1

    ## Set vorticity
    w0_p = np.power(sech((R - 0.25) / h), 2) / h
    # w0_pert_p = epsilon * params.kn*np.cos(pert_n*params.kn*X+np.pi)\
    # *(np.exp(-((Z-0.5)/sigma)**2))

    # w0_p += -np.power(sech((Z-1.5)/h), 2)/h
    # w0_pert_p += epsilon * params.kn*-np.cos(pert_n*params.kn*X+np.pi)\
    # *(np.exp(-((Z-1.5)/sigma)**2))

    w0_pert_p = epsilon * (2 * rng.random((params.nx, params.nz)) - 1.0)

    w0_p += w0_pert_p

    w.load(w0_p, is_physical=True)

    ## Set ink
    # ink0_p = np.array(np.logical_and(Z>0.5, Z<1.5)) + 1
    # ink0_p = np.array(Z > 0.5) + 1
    ink.load(w0_p, is_physical=True)


def main():
    PARAMS = {
        "nx": 2 ** 11,
        "nz": 2 ** 10,
        "lx": 16.0 / 9.0,
        "lz": 1.0,
        "Re": 1e5,
        "integrator_order": 2,
        "max_time": 3.0,
        "final_time": 10,
        "save_cadence": 0.003,
        "precision": "single",
        "spatial_derivative_order": 2,
        "integrator_order": 2,
        "integrator": "semi-implicit",
        "cfl_cutoff": 0.5,
    }
    PARAMS["initial_dt"] = 0.05 * PARAMS["lx"] / PARAMS["nx"]
    params = Parameters(PARAMS)
    params.save()

    simulation = Simulation(params, xp)

    basis_fns = [BasisFunctions.COMPLEX_EXP, BasisFunctions.COMPLEX_EXP]

    # Simulation variables
    w = simulation.make_variable("w", basis_fns)
    ink = simulation.make_variable("ink", basis_fns)

    dw = simulation.make_derivative("dw")
    dink = simulation.make_derivative("dink")

    psi = simulation.make_variable("psi", basis_fns)
    ux = simulation.make_variable("ux", basis_fns)
    uz = simulation.make_variable("uz", basis_fns)

    simulation.init_laplacian_solver(basis_fns)

    simulation.config_dump([w, ink], [dw, dink])
    simulation.config_save([w])
    simulation.config_cfl(ux, uz)
    simulation.config_scalar_trackers(
        {
            "kinetic_energy.npz": partial(
                calc_kinetic_energy, ux, uz, xp, params
            ),
        }
    )

    # Load initial conditions
    if params.load_from is not None:
        simulation.load(params.load_from)
    else:
        load_initial_conditions(params, w, ink)

    total_start = time.time()

    # Main loop
    while simulation.is_running():
        # SOLVER STARTS HERE
        calc_velocity_from_vorticity(
            w, psi, ux, uz, simulation.get_laplacian_solver()
        )

        lin_op = 1.0/params.Re * w.lap()
        dw[:] = (
            -w.vec_dot_nabla(ux.getp(), uz.getp())
        )
        simulation._integrator.integrate(w, dw, lin_op)

        # lin_op = 1.0/params.Re * ink.lap()
        # dink[:] = (
            # -ink.vec_dot_nabla(ux.getp(), uz.getp())
        # )
        # simulation._integrator.integrate(ink, dink, lin_op)

        simulation.end_loop()

    total_end = time.time() - total_start
    print(f"Total time: {total_end/3600:.2f} hr")
    print(f"Total time: {total_end:.2f} s")


if __name__ == "__main__":
    main()
