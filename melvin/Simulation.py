import numpy as np

from melvin import (
    Variable,
    DataTransferer,
    SpatialDifferentiator,
    SpectralTransformer,
    ScalarTracker,
    LaplacianSolver,
    Integrator,
    ArrayFactory,
    Timer,
    TimeDerivative,
    Ticker,
)


class Simulation:
    def __init__(self, params, xp):
        self._params = params
        self._xp = xp

        # Algorithms
        self._data_trans = DataTransferer(xp)
        self._integrator = Integrator(params, xp)
        self._array_factory = ArrayFactory(params, xp)
        self._spatial_diff = SpatialDifferentiator(
            params, xp, self._array_factory
        )
        self._spectral_trans = SpectralTransformer(
            params, xp, self._array_factory
        )

        self._t = 0
        self._loop_counter = 0

        self._dump_vars = []
        self._dump_dvars = []
        self._save_vars = []

        self._tickers = []

        self._timer = Timer()
        self._wallclock_remaining = 0.0
        self._wallclock_ticker = Ticker(
            100, self.calc_time_remaining, is_loop_counter=True
        )
        self.register_ticker(self._wallclock_ticker)

    def make_variable(self, name):
        return Variable(
            self._params,
            self._xp,
            sd=self._spatial_diff,
            st=self._spectral_trans,
            dt=self._data_trans,
            array_factory=self._array_factory,
            dump_name=name,
        )

    def make_derivative(self, name):
        return TimeDerivative(self._params, self._xp, dump_name=name)

    def init_laplacian_solver(self, basis_fns):
        self._laplacian_solver = LaplacianSolver(
            self._params,
            self._xp,
            basis_fns,
            spatial_diff=self._spatial_diff,
            array_factory=self._array_factory,
        )

    def config_dump(self, variables, derivatives):
        self._dump_vars = variables
        self._dump_dvars = derivatives

        self._dump_ticker = Ticker(
            self._params.dump_cadence,
            self.dump,
            dump_name="dump_ticker",
            is_loop_counter=False,
        )
        self.register_ticker(self._dump_ticker)

    def config_save(self, variables):
        self._save_vars = variables

        self._save_ticker = Ticker(
            self._params.save_cadence,
            lambda _t, _l: self.save(),
            dump_name="save_ticker",
            is_loop_counter=False,
        )
        self.register_ticker(self._save_ticker)

    def config_cfl(self, ux, uz):
        self._ux = ux
        self._uz = uz

        self._cfl_ticker = Ticker(
            self._params.cfl_cadence,
            self.set_dt,
            dump_name="cfl_ticker",
            is_loop_counter=True,
        )
        self.register_ticker(self._cfl_ticker)

    def config_scalar_trackers(self, trackers):
        # Setup trackers
        self._tracker_fns = [trackers[fname] for fname in trackers]
        self._trackers = [
            ScalarTracker(self._params, self._xp, fname) for fname in trackers
        ]

        # Set when tracker is calculated
        self._tracker_ticker = Ticker(
            self._params.tracker_cadence,
            self.track_scalars,
            dump_name="tracker_ticker",
            is_loop_counter=True,
        )
        self.register_ticker(self._tracker_ticker)

        # Set when tracker is saved to file
        self._save_vars += self._trackers

    def track_scalars(self, counter, cadence):
        for tracker, func in zip(self._trackers, self._tracker_fns):
            tracker.append(self._t, func())

    def set_dt(self, counter, cadence):
        self._integrator.set_dt(self._ux, self._uz)

    def save(self):
        self.print_info()
        for var in self._save_vars:
            var.save()

    def print_info(self):
        print(
            f"{self._t/self._params.final_time *100:.2f}% complete",
            f"t = {self._t:.2f}",
            f"dt = {self._integrator._dt:.2e}",
            f"Remaining: {self._wallclock_remaining/3600:.2f} hr",
        )

    def form_dumpname(self, index):
        return f"dump{index:04d}.npz"

    def dump(self, counter, cadence):
        index = counter
        fname = self.form_dumpname(index)
        dump_data = {
            var.get_name(): self._data_trans.to_host(var[:])
            for var in self._dump_vars
        }
        dump_data.update(
            {
                dvar.get_name(): self._data_trans.to_host(dvar.get_all())
                for dvar in self._dump_dvars
            }
        )
        ticker_data = {
            ticker.get_name(): ticker.dump() for ticker in self._tickers
        }
        np.savez(
            fname,
            **dump_data,
            **ticker_data,
            curr_idx=self._dump_dvars[0].get_curr_idx(),
            dt=self._integrator._dt,
            t=self._t,
            loop_counter=self._loop_counter,
            params=self._params,
        )
        self.print_info

    def load(self, index):
        fname = self.form_dumpname(index)
        dump_arrays = self._xp.load(fname)
        for var in self._dump_vars:
            var.load(dump_arrays[var.dump_name])
        for dvar in self._dump_dvars:
            dvar.load(dump_arrays[dvar.dump_name])
            dvar.set_curr_idx(dump_arrays["curr_idx"])
        for ticker in self._tickers:
            ticker.restore(dump_arrays[ticker.dump_name])
        self._integrator._dt = dump_arrays["dt"]
        self._t = dump_arrays["t"]
        self._loop_counter = dump_arrays["loop_counter"]
        # old_params = dump_arrays["params"]
        # TODO compare new params to old params

    def register_ticker(self, ticker):
        self._tickers.append(ticker)

    def end_loop(self):
        self._loop_counter += 1
        self._t += self._integrator._dt
        for ticker in self._tickers:
            ticker.tick(self._t, self._loop_counter)

    def calc_time_remaining(self, counter, cadence):
        self._timer.split()
        wallclock_per_timestep = self._timer.diff / cadence
        self._wallclock_remaining = (
            wallclock_per_timestep
            * (self._params.final_time - self._t)
            / self._integrator._dt
        )

    def get_laplacian_solver(self):
        return self._laplacian_solver
