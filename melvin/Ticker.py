class Ticker:
    def __init__(self, cadence, fn, dump_name="", is_loop_counter=True):
        self._cadence = cadence
        self._fn = fn
        self._counter = 0
        self._dump_name = dump_name

        if is_loop_counter:
            self.tick = self.__tick_counter
        else:
            self.tick = self.__tick_time

    def __tick_counter(self, _t, loop_counter):
        if self._counter < loop_counter:
            self._counter += self._cadence
            self._fn(self._counter, self._cadence)

    def __tick_time(self, t, _loop_counter):
        if self._counter < t:
            self._counter += self._cadence
            self._fn(self._counter, self._cadence)

    def dump(self):
        return {"counter": self._counter}

    def restore(self, data):
        self._counter = data["counter"]

    def get_name(self):
        return self._dump_name
