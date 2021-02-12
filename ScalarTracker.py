import numpy as np

class ScalarTracker:
    """Tracks energy through time"""
    _values = []

    def __init__(self, params, xp):
        self._xp = xp
        self._params = params

    def append(self, t, val):
        self._values += [(t, val)]

    def save(self, filename):
        """Saves to file"""
        np.save(filename, np.array(self._values))
