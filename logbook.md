# Logbook

## 2021-02-04

with NX = 4^5, NZ=2^10, dt=1e-4, Re=1e4 is suitably resolved. This produces some nice pictures.

with NX = 4^5, NZ=2^10, dt=1e-4, Re=1e5 is NOT suitably resolved. There is ringing.

## 2021-02-14

Working  on double diffusive convection. Tried with no suppression of horizontal flows and that went *poorly*. a kind of periodic convection cell occurs just as in Glatzmaier. I suspect I may have to suppress vertical flows as well. 

Crap, I think I've suppressed the horizontal flows accidentally. We want $\omega_{0m} = 0$ to suppress the x-averaged flow.

I'm curious as to whether the z-averaged temp and salinity are suppressed in Stern/Radko/Stellmach. I think they could be, to avoid accidental intrusions developing, but maybe they are what forms the staircases in the first place.

With only the horizontal x-averaged flows suppressed, the salt fingers saturate quite quickly at $t=35$.

With 
```
PARAMS = {
    "nx": 2**9,
    "nz": 2**10,
    "lx": 335.0,
    "lz": 536.0,
    "initial_dt": 1e-3,
    "Pr":7,
    "R":1.1,
    "tau":1.0/3.0,
    "final_time": 800,
    "spatial_derivative_order": 2,
    "integrator_order": 2,
    "integrator": "semi-implicit",
    "dump_cadence": 1
}
```
The simulation crashes at $t=195$. There is clearly a strange error visible from $t=190$. Not sure what the problem is. Trying with higher nx.

Still with nx=2^10 we're getting a crash at approx t=200. There are *definitely* gravity waves appearing though so that's a plus.

Trying slightly lower nx (2^9 again), upping integrator and spatial deriv order to 4 each and suppressing vertical flows. Unsure if it will help with the crash.

Seems to have solved the crashing, I bet it was the vertical flows... Runs fully to 800 but doesn't seem to transition into layers (at least looking at the KE). Looking at temperature slices too, it certainly hasn't transitioned into slices... Shame. The greater accuracy seems to have additionally helped improve the ringing in xi. This data has been archived in data/staircase_attempt1. The params are:
PARAMS = {
        "nx": 2**9,
        "nz": 2**10,
        "lx": 335.0,
        "lz": 536.0,
        "initial_dt": 1e-3,
        "Pr":7,
        "R":1.1,
        "tau":1.0/3.0,
        "final_time": 800,
        "spatial_derivative_order": 4,
        "integrator_order": 4,
        "integrator": "semi-implicit",
        "dump_cadence": 1
    }
