class Ticker:
    def __init__(self, cadence, fn, dump_name="", is_loop_counter=True):
        self._cadence = cadence
        self._fn = fn
        self._counter = 0
        self.times_fired = 0
        self._dump_name = dump_name

        if is_loop_counter:
            self.tick = self.__tick_counter
        else:
            self.tick = self.__tick_time

    def __tick_counter(self, _t, loop_counter):
        if self._counter < loop_counter:
            self.__fire()

    def __tick_time(self, t, _loop_counter):
        if self._counter < t:
            self.__fire()

    def __fire(self):
        self._fn(self)
        self._counter += self._cadence
        self.times_fired += 1

    def dump(self):
        return {"counter": self._counter, "times_fired": self.times_fired}

    def restore(self, data):
        self._counter = data["counter"]
        self.times_fired = data["times_fired"]

    def get_name(self):
        return self._dump_name
