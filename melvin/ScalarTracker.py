import numpy as np


class ScalarTracker:
    """Tracks a scalar through time"""

    def __init__(self, params, xp, filename):
        self._xp = xp
        self._params = params
        self._filename = filename
        self._values = []
        self._times = []

    def append(self, t, val):
        self._times += [t]
        self._values += [val]

    def save(self):
        """Saves to file"""
        t = self._xp.array(self._times)
        vals = self._xp.array(self._values)
        self._xp.savez(self._filename, t=t, values=vals)

    # def load(self):
    # """Loads from file"""
    # data = self._xp.load(self._filename)
    # self._times = data['t'].to_arr()
    # self._values = data['values'].to_arr()
