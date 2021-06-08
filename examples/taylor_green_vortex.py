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


def load_initial_conditions(params, w):
    x = np.linspace(0, params.lx, params.nx, endpoint=False)
    z = np.linspace(0, params.lz, params.nz, endpoint=False)
    X, Z = np.meshgrid(x, z, indexing="ij")

    w0_p = -2 * np.cos(X) * np.cos(Z)

    w.load(w0_p, is_physical=True)


def main():
    PARAMS = {
        "nx": 2 ** 10,
        "nz": 2 ** 10,
        "lx": 2 * np.pi,
        "lz": 2 * np.pi,
        "nu": 0.25,
        "initial_dt": 1e-3,
        "precision": "single",
        "spatial_derivative_order": 2,
        "integrator_order": 2,
        "integrator": "semi-implicit",
        "cfl_cutoff": 0.5,
    }
    PARAMS["final_time"] = 1000 * PARAMS["initial_dt"]
    params = Parameters(PARAMS)
    params.save()

    simulation = Simulation(params, xp)

    basis_fns = [BasisFunctions.COMPLEX_EXP, BasisFunctions.COMPLEX_EXP]

    # Simulation variables
    w = simulation.make_variable("w", basis_fns)

    dw = simulation.make_derivative("dw")

    psi = simulation.make_variable("psi", basis_fns)
    ux = simulation.make_variable("ux", basis_fns)
    uz = simulation.make_variable("uz", basis_fns)

    simulation.init_laplacian_solver(basis_fns)

    simulation.config_dump([w], [dw])
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
        load_initial_conditions(params, w)

    total_start = time.time()

    # Main loop
    while simulation.is_running():
        # SOLVER STARTS HERE
        calc_velocity_from_vorticity(
            w, psi, ux, uz, simulation.get_laplacian_solver()
        )

        lin_op = params.nu * w.lap()
        dw[:] = -w.vec_dot_nabla(ux.getp(), uz.getp())
        simulation._integrator.integrate(w, dw, lin_op)

        simulation.end_loop()

    final_ke = calc_kinetic_energy(ux, uz, xp, params)
    initial_ke = simulation._trackers[0]._values[0]
    viscous_factor = final_ke / initial_ke
    true_factor = np.exp(-simulation._t)
    print("abs error = ", abs(viscous_factor - true_factor))
    print(
        "rel error = ",
        abs(viscous_factor - true_factor) / true_factor * 100,
        "%",
    )

    total_end = time.time() - total_start
    print(f"Total time: {total_end/3600:.2f} hr")
    print(f"Total time: {total_end:.2f} s")


if __name__ == "__main__":
    main()
