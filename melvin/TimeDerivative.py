from melvin.Variable import scale_variable


class TimeDerivative:
    """
    Represents a sequence of derivatives in time
    """

    def __init__(self, params, xp, array_factory=None):
        self._params = params
        self._xp = xp
        self._curr_idx = 0
        self._data = xp.zeros(
            (
                params.integrator_order,
                params.spectral_shape[0],
                params.spectral_shape[1],
            ),
            dtype=params.complex,
        )

    def __setitem__(self, index, value):
        self._data[self._curr_idx, index] = value

    def __getitem__(self, index):
        return self._data[self._curr_idx, index]

    def get_curr_idx(self):
        return self._curr_idx

    def set_curr_idx(self, curr_idx):
        self._curr_idx = curr_idx

    def advance(self):
        self._curr_idx = (self._curr_idx + 1) % self._params.integrator_order

    def get(self, idx=0):
        return self._data[self._curr_idx + idx]

    def get_all(self):
        return self._data

    def set(self, data, idx=0):
        self._data[self._curr_idx + idx] = data

    def load(self, data):
        if isinstance(data, str):
            self.__load_from_file(data)
        else:
            self.__load_from_array(data)

    def __load_from_file(self, data):
        raise NotImplementedError

    def __load_from_array(self, data):
        # Assume data already on host
        nn, nm = self._params.nn, self._params.nm
        print(data.shape)
        for i in range(self._params.integrator_order):
            if data[i].shape != (nn, nm):
                data[i] = scale_variable(data[i], (nn, nm), self._xp)
            self._data[i] = data[i]
