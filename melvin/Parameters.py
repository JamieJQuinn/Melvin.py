import json
import numpy as np
import sys


class Parameters:
    required_params = ["nx", "nz", "lx", "lz", "final_time"]

    # Default parameters
    integrator_order = 2
    integrator = "semi-implicit"
    spatial_derivative_order = 2
    alpha = 0.51
    cfl_cutoff = 0.5
    cfl_cadence = 10  # Number of timesteps between CFL checks
    tracker_cadence = (
        100  # Number of timesteps between calculating time series
    )
    save_cadence = 100  # Time between saves
    load_from = None

    discretisation = ["spectral", "spectral"]
    precision = "double"

    # Required parameters
    nx = None
    nz = None
    lx = None
    lz = None
    final_time = None

    def __init__(self, params, validate=True):
        if validate:
            if not self.is_valid(params):
                sys.exit(-1)

        self.load_from_dict(params)

        self._original_params = params

    def load_from_dict(self, params):
        for key in params:
            setattr(self, key, params[key])
        self.set_derived_params(params)

    def is_valid(self, params):
        valid = True
        for key in self.required_params:
            if key not in params:
                print(key, "missing from input parameters.")
                valid = False
        if "fdm" in self.discretisation and self.integrator == "explicit":
            print("FDM and implicit method currently not supported.")
            valid = False
        return int(valid)

    def is_fully_spectral(self):
        return (
            self.discretisation[0] == "spectral"
            and self.discretisation[1] == "spectral"
        )

    def set_derived_params(self, params):
        if "dump_cadence" not in params:
            self.dump_cadence = 0.1 * self.final_time

        if self.discretisation[0] == "spectral":
            self.nn = (self.nx - 1) // 3
        if self.discretisation[1] == "spectral":
            self.nm = (self.nz - 1) // 3

        if self.is_fully_spectral():
            self.spectral_shape = (2 * self.nn + 1, self.nm)
        else:
            if self.discretisation[0] == "fdm":
                self.spectral_shape = (self.nx, self.nm)
            elif self.discretisation[1] == "fdm":
                self.spectral_shape = (self.nn, self.nz)
        self.physical_shape = (self.nx, self.nz)

        self.dx = self.lx / self.nx
        self.dz = self.lz / self.nz

        if self.precision == "double":
            self.complex = np.complex128
            self.float = np.float64
        elif self.precision == "single":
            self.complex = np.complex64
            self.float = np.float32

        if "initial_dt" not in params:
            self.initial_dt = 0.2 * min(self.dx, self.dz)

    def save(self, fname="params.json"):
        with open(fname, "w") as fp:
            json.dump(self._original_params, fp)
