import numpy as np

class ScalarTracker:
    """Tracks energy through time"""
    _values = []
    _times = []

    def __init__(self, params, xp, filename):
        self._xp = xp
        self._params = params
        self._filename = filename

    def append(self, t, val):
        self._times += [t]
        self._values += [val]

    def save(self):
        """Saves to file"""
        np.save(self._filename, np.array((self._times, self._values)))
