from BasisFunctions import BasisFunctions

class SpectralTransformer:
    def __init__(self, params, xp, array_factory):
        self._p = params
        self._xp = xp
        self._array_factory = array_factory

        p = self._p

        self._supscaled = self._array_factory.make_spectral(p.nx, p.nz//2+1)

    def _upscale(self, in_arr, out):
        p = self._p
        out[:p.nn+1, :p.nm] = in_arr[:p.nn+1,:]
        out[-p.nn: , :p.nm] = in_arr[-p.nn: ,:]

    def _downscale(self, in_arr, out):
        p = self._p
        out[:p.nn+1, :p.nm] = in_arr[:p.nn+1, :p.nm]
        out[-p.nn: , :p.nm] = in_arr[-p.nn: , :p.nm]

    def to_physical(self, in_arr, out=None):
        p = self._p
        if out is None:
            out = self._array_factory.make_physical()

        self._upscale(in_arr, self._supscaled)
        out[:] = self._xp.fft.irfft2(self._supscaled)*p.nx*p.nz

        return out

    def to_spectral(self, in_arr, out=None, basis_functions=[BasisFunctions.COMPLEX_EXP, BasisFunctions.COMPLEX_EXP]):
        p = self._p
        if out is None:
            out = self._array_factory.make_spectral()

        new_arr = in_arr
        x_factor = p.nx
        z_factor = p.nz

        if basis_functions[0] is BasisFunctions.COSINE:
            new_arr = self._xp.concatenate((new_arr[:-1], new_arr[:0:-1]))
            x_factor = p.nx-1

        if basis_functions[1] is BasisFunctions.COSINE:
            new_arr = self._xp.concatenate((new_arr[:,:-1], new_arr[:,:0:-1]), axis=1)
            z_factor = p.nz-1

        fft_result = self._xp.fft.rfft2(new_arr)/(x_factor*z_factor)

        if basis_functions[0] is BasisFunctions.COSINE and basis_functions[1] is BasisFunctions.COSINE:
            fft_result /= 2

        self._downscale(fft_result, out)
        return out
