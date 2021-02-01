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

        self._sdata = xp.zeros((p.nn, p.nm), dtype=p.complex)
        self._pdata = xp.zeros((p.nx, p.nz), dtype=p.float)

    def set_spectral(self, data):
        """Setter for spectral data"""
        self._sdata[:,:] = data[:,:]

    def set_physical(self, data):
        """Setter for physical data"""
        self._pdata[:,:] = data[:,:]

    def to_physical(self):
        """Convert spectral data to physical"""
        self._st.to_physical(self._sdata, self._pdata)

    def to_spectral(self):
        """Convert physical data to spectral"""
        self._st.to_physical(self._pdata, self._sdata)

    def ddx(self, out_arr):
        """Calculate spatial derivative of physical data"""
        var = self._pdata
        p = self._params

        out_arr[1:-1] = (var[2:] - var[:-2])/(2*p.dx)
        out_arr[0, :] = (var[1, :] - var[-1, :])/(2*p.dx)
        out_arr[-1, :] = (var[0, :] - var[-2, :])/(2*p.dx)

    def ddz(self, out_arr):
        """Calculate spatial derivative of physical data"""
        var = self._pdata
        p = self._params

        out_arr[:,1:-1] = (var[:,2:] - var[:,:-2])/(2*p.dz)
        out_arr[:,0] = (var[:,1] - var[:,-1])/(2*p.dz)
        out_arr[:,-1] = (var[:,0] - var[:,-2])/(2*p.dz)
