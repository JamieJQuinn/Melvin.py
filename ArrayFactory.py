class ArrayFactory:
    """A utility class for creating arrays of the correct size and shape"""
    def __init__(self, params, xp):
        self._p = params
        self._xp = xp

        if params.is_fully_spectral():
            self.make_spectral = self._make_all_spectral
        else:
            self.make_spectral = self._make_semi_spectral

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

    def _make_semi_spectral(self, ni=None, nj=None):
        """Create array representing spectral data
        :param ni: Number of elements in x direction,
        defaults to value specified in parameters
        :type ni: int, optional
        :param nj: Number of elements in z direction,
        defaults to value specified in parameters
        :type nj: int, optional
        """

        # TODO refactor this and _make_all_spectral

        if ni is None:
            ni = self._p.nn
        if nj is None:
            nj = self._p.nz
        dtype = self._p.complex

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
