class SpectralTransformer:
    def __init__(self, params, xp):
        self._p = params
        self._xp = xp

        p = self._p

        self._supscaled = self._xp.zeros((p.nx, int(p.nz/2)+1), dtype=p.complex)

    def _upscale(self, in_arr, out_arr):
        p = self._p
        out_arr[:p.nn+1, :p.nm] = in_arr[:p.nn+1,:]
        out_arr[-p.nn: , :p.nm] = in_arr[-p.nn: ,:]

    def _downscale(self, in_arr, out_arr):
        p = self._p
        out_arr[:p.nn+1, :p.nm] = in_arr[:p.nn+1, :p.nm]
        out_arr[-p.nn: , :p.nm] = in_arr[-p.nn: , :p.nm]

    def to_physical(self, in_arr, out_arr):
        p = self._p
        self._upscale(in_arr, self._supscaled)
        result = self._xp.fft.irfft2(self._supscaled)*p.nx*p.nz
        out_arr[:] = result[:]

    def to_spectral(self, in_arr, out_arr):
        p = self._p
        fft_result = self._xp.fft.rfft2(in_arr)/(p.nx*p.nz)
        self._downscale(fft_result, out_arr)
