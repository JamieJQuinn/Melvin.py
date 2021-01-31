import numpy as np

class Parameters:
    def __init__(self, PARAMS):
        self.nx = PARAMS['nx']
        self.nz = PARAMS['nz']

        self.lx = PARAMS['lx']
        self.lz = PARAMS['lz']

        self.dt = PARAMS['dt']

        self.complex = np.complex128
        self.float = np.float64

        self.set_derived_params()

    def set_derived_params(self):
        self.nn = int((self.nx-1)/3)
        self.nm = int((self.nz-1)/3)
        self.kn = 2*np.pi/self.lx
        self.km = 2*np.pi/self.lz
        self.dx = self.lx/self.nx
        self.dz = self.lz/self.nz
