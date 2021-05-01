class ArrayFactory:
    """A utility class for creating arrays of the correct size and shape"""
    def __init__(self, params, xp):
        self._p = params
        self._xp = xp

    def make_mode_number_matrices(self):
        params = self._p

        if params.is_fully_spectral():
            n = self._xp.concatenate((self._xp.arange(0, params.nn+1),  self._xp.arange(-params.nn, 0)))
            m = self._xp.arange(0, params.nm)
        elif params.discretisation[0] == 'fdm':
            n = self._xp.arange(0, params.nx)
            m = self._xp.arange(0, params.nm)
        elif params.discretisation[1] == 'fdm':
            n = self._xp.arange(0, params.nn)
            m = self._xp.arange(0, params.nz)

        return self._xp.meshgrid(n, m, indexing='ij')

    def make_spectral(self, ni=None, nj=None):
        """Create array representing spectral data
        :param ni: Number of elements in x direction,
        defaults to value specified in parameters
        :type ni: int, optional
        :param nj: Number of elements in z direction,
        defaults to value specified in parameters
        :type nj: int, optional
        """
        if ni is None:
            ni = self._p.spectral_shape[0]
        if nj is None:
            nj = self._p.spectral_shape[1]
        dtype = self._p.complex # TODO change this when dealing with e.g. sine basis functions. Complex isn't needed in that case

        return self._xp.zeros((ni, nj), dtype=dtype)

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
