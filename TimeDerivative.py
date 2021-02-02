class TimeDerivative:
    """
    Represents a sequence of derivatives in time
    """
    def __init__(self, params, xp):
        self._params = params
        self._curr_idx = 0
        self._data = xp.zeros((params.integrator_order, 2*params.nn+1, params.nm), dtype=params.complex)

    def __setitem__(self, index, value):
        self._data[self._curr_idx, index] = value

    def __getitem__(self, index):
        return self._data[self._curr_idx, index]

    def advance(self):
        self._curr_idx = (self._curr_idx + 1)%self._params.integrator_order

    def get(self, idx=0):
        return self._data[self._curr_idx+idx]

    def set(self, data, idx=0):
        self._data[self._curr_idx+idx] = data
