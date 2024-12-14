#!/usr/bin/env python3

from functools import partial
import numpy as np

import cupy
import time

from melvin import Parameters, Simulation, BasisFunctions

from melvin.utility import (
    calc_kinetic_energy,
    calc_velocity_from_vorticity,
    init_var_with_noise,
)

xp = cupy


def load_initial_conditions(params, w, tmp):
    epsilon = 1e-2

    x = np.linspace(0, params.lx, params.nx, endpoint=False)
    z = np.linspace(0, params.lz, params.nz, endpoint=False)
    X, Z = np.meshgrid(x, z, indexing="ij")

    tmp_p = 1 - Z + epsilon*(np.sin(np.pi*X/2.44))

    tmp.load(tmp_p, is_physical=True)

    init_var_with_noise(w, epsilon)
    # init_var_with_noise(tmp, epsilon)


def main():
    PARAMS = {
        "nx": 64,
        "nz": 13,
        "lx": 2.44,
        "lz": 1.0,
        "initial_dt": 1e-6,
        "cfl_cutoff": 0.5,
        "Pr": 0.5,
        "Ra": 1e6,
        "final_time": 0.05,
        "spatial_derivative_order": 2,
        "integrator_order": 2,
        "integrator": "explicit",
        "save_cadence": 1e-4,
        "dump_cadence": 1e-1,
        "discretisation": ["spectral", "fdm"],
        "precision": "single",
    }
    # PARAMS['final_time'] = 10*PARAMS['initial_dt']
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
    while simulation.is_running():
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

        # Boundary conditions
        if params.spatial_derivative_order == 2:
            w[1:, 0] = 0.0
            w[1:, -1] = 0.0

            psi[1:, 0] = 0.0
            psi[1:, -1] = 0.0

            tmp[0, 0] = 1.0
            tmp[0, -1] = 0.0

            tmp[1:, 0] = 0.0
            tmp[1:, -1] = 0.0
        elif params.spatial_derivative_order == 4:
            w[1:, :2] = 0.0
            w[1:, -2:] = 0.0

            psi[1:, :2] = 0.0
            psi[1:, -2:] = 0.0

            tmp[0, :2] = 1.0
            tmp[0, -2:] = 0.0

            tmp[1:, :2] = 0.0
            tmp[1:, -2:] = 0.0
        else:
            raise NotImplementedError("Spatial derivative order not correctly set")

        # Suppress large-scale horizontal flows
        psi[0, :] = 0.0
        w[0, :] = 0.0


        simulation.end_loop()

    total_end = time.time() - total_start
    print(f"Total time: {total_end/3600:.2f} hr")
    print(f"Total time: {total_end:.2f} s")


if __name__ == "__main__":
    main()
