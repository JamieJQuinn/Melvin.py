class Variable:
    """
    Encapsulates the physical and spectral representations of a variable
    """
    def __init__(self, params, xp, st):
        self._params = params
        self._xp = xp
        self._st = st

        xp = self._xp
        p = self._params

        self._sdata = xp.zeros((2*p.nn+1, p.nm), dtype=p.complex)
        self._pdata = xp.zeros((p.nx, p.nz), dtype=p.float)

    def __setitem__(self, index, value):
        self._sdata[index] = value

    def __getitem__(self, index):
        return self._sdata[index]

    def set_spectral(self, data):
        """Setter for spectral data"""
        self._sdata[:,:] = data[:,:]

    def set_physical(self, data):
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

    def ddx(self, out=None):
        """Calculate spatial derivative of physical data"""
        var = self._pdata
        p = self._params

        if out is None:
            out = self._xp.zeros_like(self._pdata)

        out[1:-1] = (var[2:] - var[:-2])/(2*p.dx)
        out[0, :] = (var[1, :] - var[-1, :])/(2*p.dx)
        out[-1, :] = (var[0, :] - var[-2, :])/(2*p.dx)

        return out

    def ddz(self, out=None):
        """Calculate spatial derivative of physical data"""
        var = self._pdata
        p = self._params

        if out is None:
            out = self._xp.zeros_like(self._pdata)

        out[:,1:-1] = (var[:,2:] - var[:,:-2])/(2*p.dz)
        out[:,0] = (var[:,1] - var[:,-1])/(2*p.dz)
        out[:,-1] = (var[:,0] - var[:,-2])/(2*p.dz)

        return out
