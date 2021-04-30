import matplotlib.pyplot as plt
import numpy as np
from BasisFunctions import BasisFunctions, is_fully_spectral

class Variable:
    """
    Encapsulates the physical and spectral representations of a variable
    """
    def __init__(self, params, xp, sd=None, st=None, dt=None, array_factory=None, dump_name=None,\
                 basis_functions=[BasisFunctions.COMPLEX_EXP, BasisFunctions.COMPLEX_EXP]):
        self._params = params
        self._xp = xp
        self._st = st
        self._sd = sd
        self._dt = dt
        self._array_factory = array_factory

        self._dump_name = dump_name
        self._dump_counter = 0

        self._basis_functions = basis_functions
        if self._basis_functions[0] is BasisFunctions.COMPLEX_EXP:
            self._ddx_factor = 1j*2*np.pi/self._params.lx
        elif self._basis_functions[0] is BasisFunctions.SINE:
            self._ddx_factor = np.pi/self._params.lx
        elif self._basis_functions[0] is BasisFunctions.COSINE:
            self._ddx_factor = -np.pi/self._params.lx

        if self._basis_functions[1] is BasisFunctions.COMPLEX_EXP:
            self._ddz_factor = 1j*2*np.pi/self._params.lz
        elif self._basis_functions[1] is BasisFunctions.SINE:
            self._ddz_factor = np.pi/self._params.lz
        elif self._basis_functions[1] is BasisFunctions.COSINE:
            self._ddz_factor = -np.pi/self._params.lz

        if self._params.is_fully_spectral():
            n, m = array_factory.make_mode_number_matrices()
            self.lap = -((n*np.abs(self._ddx_factor))**2 + (m*np.abs(self._ddz_factor))**2)

        if self._params.discretisation[0] == 'spectral':
            self.sd2dx2 = self.__sd2dx2_spectral
        else:
            self.sd2dx2 = self.__sd2dx2_fdm
        if self._params.discretisation[1] == 'spectral':
            self.sd2dz2 = self.__sd2dz2_spectral
        else:
            self.sd2dz2 = self.__sd2dz2_fdm

        xp = self._xp
        p = self._params

        self._sdata = array_factory.make_spectral()
        self._pdata = array_factory.make_physical()

    def __setitem__(self, index, value):
        self._sdata[index] = value

    def __getitem__(self, index):
        return self._sdata[index]

    def sets(self, data):
        """Setter for spectral data"""
        self._sdata[:,:] = data[:,:]

    def setp(self, data):
        """Setter for physical data"""
        self._pdata[:,:] = data[:,:]

    def getp(self):
        return self._pdata

    def gets(self):
        return self._sdata

    def to_physical(self):
        """Convert spectral data to physical"""
        self._st.to_physical(self._sdata, self._pdata, self._basis_functions)

    def to_spectral(self):
        """Convert physical data to spectral"""
        self._st.to_spectral(self._pdata, self._sdata, self._basis_functions)

    def load(self, data, is_physical=False):
        if isinstance(data, str):
            self.__load_from_file(data, is_physical)
        else:
            self.__load_from_array(data, is_physical)

    def __load_from_file(self, data, is_physical):
        raise NotImplementedError

    def __load_from_array(self, data, is_physical):
        data = self._dt.from_host(data)
        if is_physical:
            self.setp(data)
            self.to_spectral()
        else:
            nn, nm = self._params.nn, self._params.nm
            if data.shape != (nn, nm):
                data = scale_variable(data, (nn, nm), self._xp)
            self.sets(data)

    def pddx(self, out=None):
        """Calculate spatial derivative of physical data"""
        return self._sd.pddx(self.getp(), out=out)

    def pddz(self, out=None):
        """Calculate spatial derivative of physical data"""
        return self._sd.pddz(self.getp(), out=out)

    def sddx(self, out=None):
        """Calculate derivative of spectral data"""
        return self._sd.sddx(self.gets(), self._ddx_factor, out=out)

    def sddz(self, out=None):
        """Calculate derivative of spectral data"""
        return self._sd.sddz(self.gets(), self._ddz_factor, out=out)

    def snabla2(self, out=None):
        """Calculate $\nabla^2$ in spectral form"""
        return self.sd2dx2() + self.sd2dz2()

    def __sd2dx2_spectral(self, out=None):
        return self._sd.sd2dx2(self.gets(), self._ddx_factor, out=out)

    def __sd2dz2_spectral(self, out=None):
        return self._sd.sd2dz2(self.gets(), self._ddz_factor, out=out)

    def __sd2dx2_fdm(self, out=None):
        return self._sd.pd2dx2(self.gets(), out=out)

    def __sd2dz2_fdm(self, out=None):
        return self._sd.pd2dz2(self.gets(), out=out)

    def vec_dot_nabla(self, ux, uz, out=None):
        if out is None:
            out = self._xp.zeros_like(self._pdata)

        self.to_physical()
        out[:] = self._sd.pddx(ux*self.getp()) + self._sd.pddz(uz*self.getp())
        return self._st.to_spectral(out, basis_functions=self._basis_functions)

    def save(self):
        fname = self._dump_name + f'{self._dump_counter:04d}.npy'
        self._dump_counter += 1

        self._xp.save(fname, self._pdata)

    def on_host(self):
        return self._dt.to_host(self.gets())

    def plot(self):
        physical_host = self._dt.to_host(self._pdata)
        plt.imshow(physical_host.T)
        plt.show()

def scale_variable(var, outsize, xp):
    # Scales array in spectral space from insize to outsize
    insize = var.shape
    outvar = xp.zeros((2*outsize[0]+1, outsize[1]))
    nx_min = min(insize[0], outsize[0])
    nz_min = min(insize[1], outsize[1])
    outvar[:nx_min+1, :nz_min] = var[:nx_min+1, :nz_min]
    outvar[-nx_min:, :nz_min] = var[-nx_min:, :nz_min]

    return outvar
