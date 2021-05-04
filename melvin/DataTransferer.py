try:
    import cupy
except ModuleNotFoundError:
    pass


class DataTransferer:
    def __init__(self, xp):
        if xp.__name__ == "numpy":
            self.to_host = self.__no_transfer
            self.from_host = self.__no_transfer
        elif xp.__name__ == "cupy":
            self.to_host = self.__cupy_to_numpy
            self.from_host = self.__numpy_to_cupy

    def __no_transfer(self, data):
        return data

    def __numpy_to_cupy(self, data):
        return cupy.asarray(data)

    def __cupy_to_numpy(self, data):
        return cupy.asnumpy(data)
