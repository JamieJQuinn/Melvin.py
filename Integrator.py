import numpy as np

class Integrator:
    def _adams_bashforth_2(self, dvar):
        return self._dt/2*(3*dvar.get() - dvar.get(-1))

    def _adams_bashforth_4(self, dvar):
        return self._dt/24*(55*dvar.get() - 59*dvar.get(-1) + 37*dvar.get(-2) - 9*dvar.get(-3))

    def _adams_moulton_2(self, dvar):
        return self._dt/2*(dvar.get() + dvar.get(-1))

    def _adams_moulton_4(self, dvar):
        return self._dt/24*(9*dvar.get() +19*dvar.get(-1) -5*dvar.get(-2) + dvar.get(-3))

    def set_dt(self, ux, uz):
        """Sets dt based on CFL limit"""
        cfl_dt = min(self._dx/self._xp.max(ux.getp()), self._dz/self._xp.max(uz.getp()))
        if self._dt > cfl_dt or np.isnan(cfl_dt):
            print("CFL condition breached")
            return
        while self._dt > self._cfl_cutoff*cfl_dt:
            self._dt = self._dt*0.9
        return self._dt

    def override_dt(self, dt):
        """Manually set dt"""
        self._dt = dt

    def get_dt(self):
        return self._dt

    def _explicit(self, var, dvar, diffusion_term):
        dvar[:] += diffusion_term
        var[:] += self.predictor(dvar)
        dvar.advance()

    def _semi_implicit_spectral(self, var, dvar, lin_op):
        alpha = self._alpha
        dt = self._dt
        RHS = (1+(1-alpha)*dt*lin_op)*var[:] + self.predictor(dvar)
        var[:] = RHS/(1-alpha*dt*lin_op)
        dvar.advance()

    def __init__(self, params, xp):
        self._dt = params.initial_dt
        self._dx = params.dx
        self._dz = params.dz
        self._cfl_cutoff = params.cfl_cutoff
        self._xp = xp
        if params.integrator_order == 2:
            self.predictor = self._adams_bashforth_2
            self.corrector = self._adams_moulton_2
        elif params.integrator_order == 4:
            self.predictor = self._adams_bashforth_4
            self.corrector = self._adams_moulton_4

        if params.integrator == "semi-implicit":
            if params.is_fully_spectral():
                self.integrate = self._semi_implicit_spectral
                self._alpha = params.alpha
        elif params.integrator == "explicit":
            self.integrate = self._explicit
