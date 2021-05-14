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


def calc_nusselt_number(tmp, uz, xp, params):
    # From Stellmach et al 2011 (DOI: 10.1017/jfm.2011.99)
    flux = xp.mean(tmp.getp() * uz.getp())
    return 1.0 - flux


def load_initial_conditions(params, w, tmp, xi):
    epsilon = 1e-2
    init_var_with_noise(w, epsilon)
    init_var_with_noise(tmp, epsilon)
    init_var_with_noise(xi, epsilon)


def main():
    factor = 0.25
    LX = 335.0 * factor
    LZ = 9.0 / 16 * LX
    PARAMS = {
        "nx": 2 ** 9,
        "nz": 2 ** 8,
        "lx": LX,
        "lz": LZ,
        "initial_dt": 1e-3,
        "cfl_cutoff": 0.5,
        "Pr": 7.0,
        "R0": 1.1,
        "tau": 1.0 / 3.0,
        "final_time": 100,
        "spatial_derivative_order": 2,
        "integrator_order": 2,
        "integrator": "semi-implicit",
        "save_cadence": 0.05,
        "dump_cadence": 10,
        "precision": "float",
    }
    params = Parameters(PARAMS)
    params.save()

    simulation = Simulation(params, xp)

    basis_fns = [BasisFunctions.COMPLEX_EXP, BasisFunctions.COMPLEX_EXP]

    # Simulation variables
    w = simulation.make_variable("w", basis_fns)
    tmp = simulation.make_variable("tmp", basis_fns)
    xi = simulation.make_variable("xi", basis_fns)

    dw = simulation.make_derivative("dw")
    dtmp = simulation.make_derivative("dtmp")
    dxi = simulation.make_derivative("dxi")

    psi = simulation.make_variable("psi", basis_fns)
    ux = simulation.make_variable("ux", basis_fns)
    uz = simulation.make_variable("uz", basis_fns)

    simulation.init_laplacian_solver(psi._basis_functions)

    simulation.config_dump([w, tmp, xi], [dw, dtmp, dxi])
    simulation.config_save([tmp])
    simulation.config_cfl(ux, uz)
    simulation.config_scalar_trackers(
        {
            "kinetic_energy.npz": partial(
                calc_kinetic_energy, ux, uz, xp, params
            ),
            "nusselt_number.npz": partial(
                calc_nusselt_number, tmp, uz, xp, params
            ),
        }
    )

    # Load initial conditions
    if params.load_from is not None:
        simulation.load(params.load_from)
    else:
        load_initial_conditions(params, w, tmp, xi)

    total_start = time.time()

    # Main loop
    while simulation._t < params.final_time:
        # SOLVER STARTS HERE
        calc_velocity_from_vorticity(
            w, psi, ux, uz, simulation.get_laplacian_solver()
        )

        lin_op = params.Pr * w.lap()
        dw[:] = (
            -w.vec_dot_nabla(ux.getp(), uz.getp())
            + params.Pr * xi.sddx()
            - params.Pr * tmp.sddx()
        )
        simulation._integrator.integrate(w, dw, lin_op)

        lin_op = tmp.lap()
        dtmp[:] = -tmp.vec_dot_nabla(ux.getp(), uz.getp()) - uz[:]
        simulation._integrator.integrate(tmp, dtmp, lin_op)

        lin_op = params.tau * xi.lap()
        dxi[:] = -xi.vec_dot_nabla(ux.getp(), uz.getp()) - uz[:] / params.R0
        simulation._integrator.integrate(xi, dxi, lin_op)

        # Remove mean z variation
        tmp[:, 0] = 0.0
        xi[:, 0] = 0.0

        simulation.end_loop()

    total_end = time.time() - total_start
    print(f"Total time: {total_end/3600:.2f} hr")
    print(f"Total time: {total_end:.2f} s")


if __name__ == "__main__":
    main()
