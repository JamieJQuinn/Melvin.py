class Integrator:
    def _adams_bashforth_2(self, dvar):
        return self._dt/2*(3*dvar.get() - dvar.get(-1))

    def _adams_bashforth_4(self, dvar):
        return self._dt/24*(55*dvar.get() - 59*dvar.get(-1) + 37*dvar.get(-2) - 9*dvar.get(-3))

    def _adams_moulton_2(self, dvar):
        return self._dt/2*(dvar.get() + dvar.get(-1))

    def _adams_moulton_4(self, dvar):
        return self._dt/24*(9*dvar.get() +19*dvar.get(-1) -5*dvar.get(-2) + dvar.get(-3))

    def set_dt(self, dt):
        self._dt = dt

    def __init__(self, params):
        self._dt = params.initial_dt
        if params.integrator_order == 2:
            self.predictor = self._adams_bashforth_2
            self.corrector = self._adams_moulton_2
        elif params.integrator_order == 4:
            self.predictor = self._adams_bashforth_4
            self.corrector = self._adams_moulton_4
