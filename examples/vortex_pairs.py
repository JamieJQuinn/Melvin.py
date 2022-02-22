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

SEED = 0
RNG = default_rng(SEED)

def calc_vortex(circ, x, z, xc, zc, a):
    # Lamb-Oseen vortex
    r2 = (x-xc)**2 + (z-zc)**2
    vortex = circ / (np.pi * a**2) * np.exp(-r2/(a**2))
    return vortex

def random(a, b):
    return (b-a)*RNG.random() + a

def load_initial_conditions(params, w, ink):
    x = np.linspace(0, params.lx, params.nx, endpoint=False)
    z = np.linspace(0, params.lz, params.nz, endpoint=False)
    X, Z = np.meshgrid(x, z, indexing="ij")

    CIRC = 1.0

    ## Set vorticity
    w0_p = 0.0 * X

    ## Add random vortices
    for i in range(50):
        circ = random(-1, 1)
        x0 = random(0, params.lx)
        z0 = random(0, params.lz)
        a = random(0.05, 0.1)
        w0_p += calc_vortex(
            circ, X, Z, x0, z0, a
        )

    # Left moving pair
    # w0_p += calc_vortex(CIRC, X, Z, params.lx/5, params.lz/2+0.1, 0.1)
    # w0_p += calc_vortex(CIRC, X, Z, params.lx/5, params.lz/2-0.1, 0.1)

    # # Right moving pair
    # w0_p += calc_vortex(CIRC, X, Z, 4*params.lx/5, params.lz/2+0.1, 0.1)
    # w0_p += calc_vortex(CIRC, X, Z, 4*params.lx/5, params.lz/2-0.1, 0.1)

    w.load(w0_p, is_physical=True)


def main():
    PARAMS = {
        "nx": 2 ** 10,
        "nz": 2 ** 9,
        "lx": 16.0 / 9.0,
        "lz": 1.0,
        "Re": 1e4,
        "final_time": 20,
        "save_cadence": 0.01,
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

        lin_op = 1.0 / params.Re * w.lap()
        dw[:] = -w.vec_dot_nabla(ux.getp(), uz.getp())
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
