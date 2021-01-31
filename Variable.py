class Variable:
    def __init__(self, params, xp, st):
        self._params = params
        self._xp = xp
        self._st = st

        xp = self._xp
        p = self._params

        self._sdata = xp.zeros((p.nn, p.nm), dtype=p.complex)
        self._pdata = xp.zeros((p.nx + 2*p.ng, p.nz + 2*p.ng), dtype=p.float)

    def set_spectral(self, data):
        self._sdata[:,:] = data[:,:]

    def set_physical(self, data):
        self._pdata[:,:] = data[:,:]

    def to_physical(self):
        self._st.to_physical(self._sdata, self.get_internal())

    def to_spectral(self):
        self._st.to_physical(self.get_internal(), self._sdata)

    def get_internal(self):
        p = self._params
        return self._pdata[p.ng:-p.ng, p.ng:-p.ng]

    def set_internal(self, value):
        p = self._params
        self._pdata[p.ng:-p.ng, p.ng:-p.ng] = value

    def apply_periodic_bcs(self):
        p = self._params
        self._pdata[:p.ng] = self._pdata[-2*p.ng:p.ng]
        self._pdata[-p.ng:] = self._pdata[p.ng:2*p.ng]

    def ddx(self, out_arr):
        var = self._pdata
        p = self._params

        out_arr[:] = (var[1+p.ng:] - var[:-p.ng-1])/(2*p.dx)

    def ddz(self, out_arr):
        var = self._pdata
        p = self._params

        out_arr[:] = (var[:,1+p.ng:] - var[:,:-p.ng-1])/(2*p.dz)
