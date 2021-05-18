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
    sech,
)

xp = cupy


def create_shear_layer(z0, width, Z):
    return -np.power(sech((Z - z0) / width), 2) / width


def create_sine_perturbation(z0, width, wavenumber, params, X, Z):
    spectral_wavenumber = 2 * np.pi / params.lx
    return (
        spectral_wavenumber
        * -np.cos(wavenumber * spectral_wavenumber * X)
        * np.exp(-(((Z - z0) / width) ** 2))
    )


def load_initial_conditions(params, w, j):
    x = np.linspace(0, params.lx, params.nx, endpoint=False)
    z = np.linspace(0, params.lz, params.nz, endpoint=False)
    X, Z = np.meshgrid(x, z, indexing="ij")

    rng = default_rng(0)

    epsilon = 0.01

    # Create a shear layer circle
    # R = np.sqrt((X - (params.lx / 2)) ** 2 + (Z - 0.5) ** 2)
    # w0_p = create_hor_shear_layer(0.25, 0.1, R)

    # Create straight shear layer
    # w0_p = create_shear_layer(0.5, 0.1, Z)
    j0_p = create_shear_layer(0.5, 0.01, Z)

    # Create sine perturbation
    # j0_p += epsilon * create_sine_perturbation(0.5, 0.2, 1, params, X, Z)

    # Create random perturbation
    j0_p += epsilon * (2 * rng.random((params.nx, params.nz)) - 1.0)

    # Load into vorticity
    j.load(j0_p, is_physical=True)


def main():
    PARAMS = {
        "nx": 2 ** 12,
        "nz": 2 ** 11,
        "lx": 16.0 / 9,
        "lz": 1.0,
        "initial_dt": 1e-4,
        "cfl_cutoff": 0.5,
        "Re": 1e6,
        "S": 1e6,
        "final_time": 1,
        "spatial_derivative_order": 2,
        "integrator_order": 2,
        "integrator": "semi-implicit",
        "save_cadence": 0.003,
        "dump_cadence": 0.1,
        "precision": "single",
    }
    params = Parameters(PARAMS)
    params.save()

    simulation = Simulation(params, xp)

    basis_fns = [BasisFunctions.COMPLEX_EXP, BasisFunctions.COMPLEX_EXP]

    # Simulation variables
    w = simulation.make_variable("w", basis_fns)
    j = simulation.make_variable("j", basis_fns)

    dw = simulation.make_derivative("dw")
    dj = simulation.make_derivative("dj")

    psi = simulation.make_variable("psi", basis_fns)
    ux = simulation.make_variable("ux", basis_fns)
    uz = simulation.make_variable("uz", basis_fns)

    phi = simulation.make_variable("phi", basis_fns)
    bx = simulation.make_variable("bx", basis_fns)
    bz = simulation.make_variable("bz", basis_fns)

    simulation.init_laplacian_solver(psi._basis_functions)

    simulation.config_dump([w, j], [dw, dj])
    simulation.config_save([j])
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
        load_initial_conditions(params, w, j)

    total_start = time.time()

    # Main loop
    while simulation.is_running():
        # SOLVER STARTS HERE
        calc_velocity_from_vorticity(
            w, psi, ux, uz, simulation.get_laplacian_solver()
        )

        # FIXME - this is not really velocity!! This should be refactored!
        calc_velocity_from_vorticity(
            j, phi, bx, bz, simulation.get_laplacian_solver()
        )

        lin_op = 1.0 / params.Re * w.lap()
        dw[:] = -w.vec_dot_nabla(ux.getp(), uz.getp()) + j.vec_dot_nabla(
            bx.getp(), bz.getp()
        )
        simulation._integrator.integrate(w, dw, lin_op)

        lin_op = 1.0 / params.S * j.lap()
        dj[:] = -j.vec_dot_nabla(ux.getp(), uz.getp()) + w.vec_dot_nabla(
            bx.getp(), bz.getp()
        )
        simulation._integrator.integrate(j, dj, lin_op)

        simulation.end_loop()

    total_end = time.time() - total_start
    print(f"Total time: {total_end/3600:.2f} hr")
    print(f"Total time: {total_end:.2f} s")


if __name__ == "__main__":
    main()
