class ArrayFactory:
    """A utility class for creating arrays of the correct size and shape"""
    def __init__(self, params, xp):
        self._p = params
        self._xp = xp

    def make_spectral(self, nn=None, nm=None):
        """Create array representing spectral data
        :param nn: Number of spectral elements in x direction,
        defaults to value specified in parameters
        :type nn: int, optional
        :param nm: Number of spectral elements in z direction,
        defaults to value specified in parameters
        :type nm: int, optional
        """
        if nn is None:
            nn = 2*self._p.nn+1
        if nm is None:
            nm = self._p.nm
        dtype = self._p.complex

        return self._xp.zeros((nn, nm), dtype=dtype)

    def make_physical(self, nx=None, nz=None):
        """Create array representing physical data
        :param nx: Number of points in x direction,
        defaults to value specified in parameters
        :type nx: int, optional
        :param nz: Number of points in z direction,
        defaults to value specified in parameters
        :type nz: int, optional
        """
        if nx is None:
            nx = self._p.nx
        if nz is None:
            nz = self._p.nz
        dtype = self._p.float

        return self._xp.zeros((nx, nz), dtype=dtype)
