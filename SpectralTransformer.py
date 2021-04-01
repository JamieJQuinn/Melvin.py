from BasisFunctions import BasisFunctions

class SpectralTransformer:
    def __init__(self, params, xp, array_factory):
        self._p = params
        self._xp = xp
        self._array_factory = array_factory

        p = self._p

    def _upscale(self, in_arr, out):
        p = self._p
        out[:p.nn+1, :p.nm] = in_arr[:p.nn+1,:]
        out[-p.nn: , :p.nm] = in_arr[-p.nn: ,:]

    def _downscale(self, in_arr, out):
        p = self._p
        out[:p.nn+1, :p.nm] = in_arr[:p.nn+1, :p.nm]
        out[-p.nn: , :p.nm] = in_arr[-p.nn: , :p.nm]

    def to_physical(self, in_arr, out=None, basis_functions=[BasisFunctions.COMPLEX_EXP, BasisFunctions.COMPLEX_EXP]):
        p = self._p
        if out is None:
            out = self._array_factory.make_physical()

        x_factor = p.nx
        z_factor = p.nz

        upscale_size = [p.nx, p.nz//2+1]

        if basis_functions[0] is BasisFunctions.COSINE:
            upscale_size[0] = 2*(p.nx-1)
            x_factor = p.nx-1
        elif basis_functions[0] is BasisFunctions.SINE:
            upscale_size[0] = 2*(p.nx-1)
            x_factor = -1j*(p.nx-1)

        if basis_functions[1] is BasisFunctions.COSINE:
            upscale_size[1] = 2*(p.nz-1)//2+1
            z_factor = p.nz-1
        elif basis_functions[1] is BasisFunctions.SINE:
            upscale_size[1] = 2*(p.nz-1)//2+1
            z_factor = -1j*(p.nz-1)

        if  basis_functions[0] is BasisFunctions.COSINE:
            in_arr[0] *= 2 # FFT double counts 0-th order term
        if basis_functions[1] is BasisFunctions.COSINE:
            in_arr[:,0] *= 2

        in_arr *= x_factor*z_factor

        upscaled = self._array_factory.make_spectral(upscale_size[0], upscale_size[1])
        self._upscale(in_arr, upscaled)
        fft_result = self._xp.fft.irfft2(upscaled)

        if basis_functions[0] is BasisFunctions.COSINE or basis_functions[0] is BasisFunctions.SINE:
            fft_result = fft_result[:p.nx] # mirror signal around end point

        if basis_functions[1] is BasisFunctions.COSINE or basis_functions[1] is BasisFunctions.SINE:
            fft_result = fft_result[:,:p.nz] # mirror signal around end point

        out[:] = fft_result

        return out

    def to_spectral(self, in_arr, out=None, basis_functions=[BasisFunctions.COMPLEX_EXP, BasisFunctions.COMPLEX_EXP]):
        p = self._p
        if out is None:
            out = self._array_factory.make_spectral()

        new_arr = in_arr
        x_factor = p.nx
        z_factor = p.nz

        if basis_functions[0] is BasisFunctions.COSINE:
            new_arr = self._xp.concatenate((new_arr[:-1], new_arr[:0:-1])) # mirror signal around end point
            x_factor = p.nx-1
        elif basis_functions[0] is BasisFunctions.SINE:
            new_arr = self._xp.concatenate((new_arr[:-1], -new_arr[:0:-1])) # mirror and flip
            x_factor = -1j*(p.nx-1)

        if basis_functions[1] is BasisFunctions.COSINE:
            new_arr = self._xp.concatenate((new_arr[:,:-1], new_arr[:,:0:-1]), axis=1)
            z_factor = p.nz-1
        elif basis_functions[1] is BasisFunctions.SINE:
            new_arr = self._xp.concatenate((new_arr[:,:-1], -new_arr[:,:0:-1]), axis=1)
            z_factor = -1j*(p.nz-1)

        fft_result = self._xp.fft.rfft2(new_arr)/(x_factor*z_factor)

        if  basis_functions[0] is BasisFunctions.COSINE:
            fft_result[0] /= 2 # FFT double counts 0-th order term
        if basis_functions[1] is BasisFunctions.COSINE:
            fft_result[:,0] /= 2

        self._downscale(fft_result, out)
        return out
