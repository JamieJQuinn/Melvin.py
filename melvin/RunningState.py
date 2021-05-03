import json

class RunningState:
    t = 0.0
    dt = 0.0
    save_counter = 0
    dump_counter = 0
    dump_index = 0
    ke_counter = 0
    cfl_counter = 0
    loop_counter = 0

    _state = {
        "t": 0.0,
        "dt": 0.0,
        "save_counter": 0,
        "dump_counter": 0,
        "dump_index": 0,
        "ke_counter": 0,
        "cfl_counter": 0,
        "loop_counter": 0
    }

    def __init__(self, params):
        if params.load_from is not None:
            # Load from saved state
            with open(self.form_dumpname(params.load_from), 'r') as fp:
                self._state = json.load(fp)
        else:
            # Set from params
            self._state['dt'] = params.initial_dt

        # Load dict state into object
        for key in self._state:
            setattr(self, key, self._state[key])

    def save(self, index):
        """Load object into dict; save as json"""
        for key in self._state:
            self._state[key] = getattr(self, key)
        with open(self.form_dumpname(index), 'w') as fp:
            json.dump(self._state, fp)

    def form_dumpname(self, index):
        return f'dump{index:04d}.json'
