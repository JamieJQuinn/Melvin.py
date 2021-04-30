from BasisFunctions import BasisFunctions

class SpectralTransformer:
    def __init__(self, params, xp, array_factory):
        self._p = params
        self._xp = xp
        self._array_factory = array_factory

        p = self._p

        if p.is_fully_spectral():
            self.to_physical = self.__to_physical_2d
            self.to_spectral = self.__to_spectral_2d
            self._scale = self._scale_2d
        else:
            self.to_physical = self.__to_physical_1d
            self.to_spectral = self.__to_spectral_1d
            self._scale = self._scale_1d

    def _scale_2d(self, in_arr, out):
        p = self._p
        out[:p.nn+1, :p.nm] = in_arr[:p.nn+1,:p.nm]
        out[-p.nn: , :p.nm] = in_arr[-p.nn: ,:p.nm]

    def _scale_1d(self, in_arr, out, axis):
        p = self._p
        if axis==0:
            out[:p.nn] = in_arr[:p.nn]
        elif axis==1:
            out[:,:p.nm] = in_arr[:,:p.nm]

    def __to_physical_1d(self, in_arr, out=None, basis_functions=[BasisFunctions.COMPLEX_EXP, BasisFunctions.COMPLEX_EXP]):
        p = self._p
        if out is None:
            out = self._array_factory.make_physical()

        if basis_functions[0] is BasisFunctions.FDM:
            axis=1
            factor = p.nz
            upscaled = self._array_factory.make_spectral(p.nx, p.nz//2+1)
        elif basis_functions[1] is BasisFunctions.FDM:
            axis=0
            factor = p.nx
            upscaled = self._array_factory.make_spectral(p.nx//2+1, p.nz)
        else:
            raise Exception("One basis function must be FDM")

        self._scale(in_arr, upscaled, axis)
        fft_result = self._xp.fft.irfft(upscaled*factor, axis=axis)
        out[:] = fft_result

        return out
    
    def __to_spectral_1d(self, in_arr, out=None, basis_functions=[BasisFunctions.COMPLEX_EXP, BasisFunctions.COMPLEX_EXP]):
        p = self._p
        if out is None:
            out = self._array_factory.make_spectral()

        if basis_functions[0] is BasisFunctions.FDM:
            axis=1
            factor = p.nz
        elif basis_functions[1] is BasisFunctions.FDM:
            axis=0
            factor = p.nx
        else:
            raise Exception("One basis function must be FDM")

        fft_result = self._xp.fft.rfft(in_arr, axis=axis)/factor
        self._scale(fft_result, out, axis)

        return out

    def __to_physical_2d(self, in_arr, out=None, basis_functions=[BasisFunctions.COMPLEX_EXP, BasisFunctions.COMPLEX_EXP]):
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
        self._scale(in_arr, upscaled)
        fft_result = self._xp.fft.irfft2(upscaled)

        if basis_functions[0] is BasisFunctions.COSINE or basis_functions[0] is BasisFunctions.SINE:
            fft_result = fft_result[:p.nx] # mirror signal around end point

        if basis_functions[1] is BasisFunctions.COSINE or basis_functions[1] is BasisFunctions.SINE:
            fft_result = fft_result[:,:p.nz] # mirror signal around end point

        out[:] = fft_result

        return out

    def __to_spectral_2d(self, in_arr, out=None, basis_functions=[BasisFunctions.COMPLEX_EXP, BasisFunctions.COMPLEX_EXP]):
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

        self._scale(fft_result, out)
        return out
