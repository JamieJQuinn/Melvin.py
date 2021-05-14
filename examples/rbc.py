#!/usr/bin/env python3

from functools import partial
import numpy as np

import cupy
import time
import matplotlib.pyplot as plt

from melvin import Parameters, Simulation, BasisFunctions

from melvin.utility import (
    calc_kinetic_energy,
    calc_velocity_from_vorticity,
    init_var_with_noise,
)

xp = cupy


def load_initial_conditions(params, w, tmp):
    epsilon = 1e-2
    init_var_with_noise(w, epsilon)
    init_var_with_noise(tmp, epsilon)


def main():
    PARAMS = {
        "nx": 2 ** 7,
        "nz": 2 ** 6,
        "lx": 16.0 / 9,
        "lz": 1.0,
        "initial_dt": 1e-5,
        "cfl_cutoff": 0.5,
        "Pr": 1.0,
        "Ra": 1e6,
        "final_time": 1e-1,
        "spatial_derivative_order": 2,
        "integrator_order": 2,
        "integrator": "explicit",
        "save_cadence": 5e-5,
        "dump_cadence": 1e-1,
        "discretisation": ["spectral", "fdm"],
        "precision": "float",
    }
    params = Parameters(PARAMS)
    params.save()

    simulation = Simulation(params, xp)

    basis_fns = [BasisFunctions.COMPLEX_EXP, BasisFunctions.FDM]

    # Simulation variables
    w = simulation.make_variable("w", basis_fns)
    tmp = simulation.make_variable("tmp", basis_fns)

    dw = simulation.make_derivative("dw")
    dtmp = simulation.make_derivative("dtmp")

    psi = simulation.make_variable("psi", basis_fns)
    ux = simulation.make_variable("ux", basis_fns)
    uz = simulation.make_variable("uz", basis_fns)

    simulation.init_laplacian_solver(psi._basis_functions)

    simulation.config_dump([w, tmp], [dw, dtmp])
    simulation.config_save([tmp])
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
        load_initial_conditions(params, w, tmp)

    total_start = time.time()

    # Main loop
    while simulation._t < params.final_time:
        # SOLVER STARTS HERE
        calc_velocity_from_vorticity(
            w, psi, ux, uz, simulation.get_laplacian_solver()
        )

        diffusion_term = params.Pr * w.snabla2()
        dw[:] = (
            -w.vec_dot_nabla(ux.getp(), uz.getp())
            - params.Pr * params.Ra * tmp.sddx()
        )
        simulation._integrator.integrate(w, dw, diffusion_term)

        diffusion_term = tmp.snabla2()
        dtmp[:] = -tmp.vec_dot_nabla(ux.getp(), uz.getp())
        simulation._integrator.integrate(tmp, dtmp, diffusion_term)

        w[1:, 0] = 0.0
        w[1:, -1] = 0.0

        psi[1:, 0] = 0.0
        psi[1:, -1] = 0.0

        psi[0, :] = 0.0
        w[0, :] = 0.0

        tmp[0, 0] = 1.0
        tmp[0, -1] = 0.0

        tmp[1:, 0] = 0.0
        tmp[1:, -1] = 0.0

        simulation.end_loop()

    total_end = time.time() - total_start
    print(f"Total time: {total_end/3600:.2f} hr")
    print(f"Total time: {total_end:.2f} s")


if __name__ == "__main__":
    main()
