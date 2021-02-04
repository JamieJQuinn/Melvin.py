import matplotlib.pyplot as plt
import numpy as np

class Variable:
    """
    Encapsulates the physical and spectral representations of a variable
    """
    def __init__(self, params, xp, st=None, dt=None, dump_name=None):
        self._params = params
        self._xp = xp
        self._st = st
        self._dt = dt
        self._dump_name = dump_name
        self._dump_counter = 0

        xp = self._xp
        p = self._params

        self._sdata = xp.zeros((2*p.nn+1, p.nm), dtype=p.complex)
        if st is not None:
            # This means we want to transform
            self._pdata = xp.zeros((p.nx, p.nz), dtype=p.float)

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
        self._st.to_physical(self._sdata, self._pdata)

    def to_spectral(self):
        """Convert physical data to spectral"""
        self._st.to_spectral(self._pdata, self._sdata)

    def load_ics(self, data, is_physical=True):
        if isinstance(data, str):
            self.__load_ics_from_file(data, is_physical)
        else:
            self.__load_ics_from_array(data, is_physical)

    def __load_ics_from_file(self, data, is_physical):
        pass

    def __load_ics_from_array(self, data, is_physical):
        data = self._dt.from_host(data)
        if is_physical:
            self.setp(data)
            self.to_spectral()
        else:
            self.sets(data)

    def ddx(self, out=None):
        """Calculate spatial derivative of physical data"""
        var = self._pdata
        p = self._params

        if out is None:
            out = self._xp.zeros_like(self._pdata)

        # out[1:-1] = (var[2:] - var[:-2])/(2*p.dx)
        # out[0, :] = (var[1, :] - var[-1, :])/(2*p.dx)
        # out[-1, :] = (var[0, :] - var[-2, :])/(2*p.dx)

        out[2:-2] = (-0.25*var[4:] + 2*var[3:-1] - 2*var[1:-3] + 0.25*var[:-4])/(3*p.dz)
        out[:2] = (-0.25*var[2:4] + 2*var[1:3] - 2*var.take((-1, 0), axis=0) + 0.25*var[-2:])/(3*p.dz)
        out[-2:] = (-0.25*var[:2] + 2*var.take((-1, 0), axis=0) - 2*var[-3:-1] + 0.25*var[-4:-2])/(3*p.dz)

        return out

    def ddz(self, out=None):
        """Calculate spatial derivative of physical data"""
        var = self._pdata
        p = self._params

        if out is None:
            out = self._xp.zeros_like(self._pdata)

        # out[:,1:-1] = (var[:,2:] - var[:,:-2])/(2*p.dz)
        # out[:,0] = (var[:,1] - var[:,-1])/(2*p.dz)
        # out[:,-1] = (var[:,0] - var[:,-2])/(2*p.dz)

        out[:,2:-2] = (-0.25*var[:,4:] + 2*var[:,3:-1] - 2*var[:,1:-3] + 0.25*var[:,:-4])/(3*p.dz)
        out[:,:2] = (-0.25*var[:,2:4] + 2*var[:,1:3] - 2*var.take((-1, 0), axis=1) + 0.25*var[:,-2:])/(3*p.dz)
        out[:,-2:] = (-0.25*var[:,:2] + 2*var.take((-1, 0), axis=1) - 2*var[:,-3:-1] + 0.25*var[:,-4:-2])/(3*p.dz)

        return out

    def vec_dot_nabla(self, vx, vz, out=None):
        if out is None:
            out = self._xp.zeros_like(self._pdata)

        p = self._params

        out[:] = vx*self.ddx() + vz*self.ddz()
        return self._st.to_spectral(out)

    def save(self):
        physical_host = self._dt.to_host(self._pdata)
        fname = self._dump_name + '{:04d}'.format(self._dump_counter) + ".npy"
        self._dump_counter += 1

        np.save(fname, self._pdata)

    def plot(self):
        physical_host = self._dt.to_host(self._pdata)
        plt.imshow(physical_host.T)
        plt.show()
