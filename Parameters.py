import numpy as np
import sys

class Parameters:
    nx = 1
    nz = 1
    lx = 1
    lz = 1
    dt = 1
    dump_cadence = 1
    Re = 1
    integrator_order = 2
    spatial_derivative_order = 2
    alpha = 1.01

    def __init__(self, PARAMS, validate=True):
        self.nx = PARAMS['nx']
        self.nz = PARAMS['nz']

        if validate:
            if not self.is_valid(PARAMS):
                sys.exit(-1)

        if 'lx' in PARAMS:
            self.lx = PARAMS['lx']

        if 'lz' in PARAMS:
            self.lz = PARAMS['lz']

        if 'dt' in PARAMS:
            self.dt = PARAMS['dt']

        if 'max_time' in PARAMS:
            self.max_time = PARAMS['max_time']

        if 'dump_cadence' in PARAMS:
            self.dump_cadence = PARAMS['dump_cadence']

        if 'Re' in PARAMS:
            self.Re = PARAMS['Re']

        if 'integrator_order' in PARAMS:
            self.integrator_order = PARAMS['integrator_order']

        if 'spatial_derivative_order' in PARAMS:
            self.spatial_derivative_order = PARAMS['spatial_derivative_order']

        self.complex = np.complex128
        self.float = np.float64

        self.set_derived_params()

    def is_valid(self, PARAMS):
        for key in ['lx', 'lz', 'dt', 'max_time', 'dump_cadence']:
            if key not in PARAMS:
                print(key, "missing from input parameters.")
                return 0
        return 1

    def set_derived_params(self):
        self.nn = int((self.nx-1)/3)
        self.nm = int((self.nz-1)/3)
        self.kn = 2*np.pi/self.lx
        self.km = 2*np.pi/self.lz
        self.dx = self.lx/self.nx
        self.dz = self.lz/self.nz
