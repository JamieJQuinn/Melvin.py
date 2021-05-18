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

YASSSS it worked! Then crashed... The outputs are in staircase_attempt2. Curious about the crash, I wonder if it was the lack of resolution in the x-dir. I'll try upping it. I've also implemented saving and loading from a dump file.

# 2021-02-16

Running some speed tests with params
    PARAMS = {
        "nx": 2**9,
        "nz": 2**9,
        "lx": 335.0,
        "lz": 536.0,
        "initial_dt": 1e-3,
        "cfl_cutoff": 0.1,
        "Pr":7.0,
        "R":1.1,
        "tau":1.0/3.0,
        "final_time": 10,
        "spatial_derivative_order": 2,
        "integrator_order": 2,
        "integrator": "explicit",
        "save_cadence": 11,
        "dump_cadence": 50
    }

time = 48s
with semi-implicit; time = 47s
with 4th order space; time = 78s
also with 4th order time; time = 82s
4th order space but periodic boundary removed; time = 62s <- This is significant!
4th order space but nonlinear advection term is nabla cdot vT rather than v \cdot nabla T; time = 78s
Passing slice to FFT; time = 85s

## 2021-02-20

So I'm not convinced the length of time taken to reach the final staircase is related to the suppression of the x- or z-averaged velocities. Plotting the x-modes of the z-averaged vorticity shows a massive spike around n=29. This is not an effect of aliasing (tried setting nx = 4*nn) and does not seem to be an effect of the CFL condition (although this is the most likely culprit). The reason I say that is that the timestep just keeps getting lower; it doesn't stabilise at any point... This is also true for higher resolutions. Reducing tau does not help, it crashes later but the buildup of energy in mode 29 still occurs.

Trying artificially increasing the viscosity applied to the vorticity. This did nothing. The run that didn't seem to start glitching was a slightly smaller box using these settings:
    PARAMS = {
        "nx": 2**8,
        "nz": 2**8,
        # "lx": 335.0,
        # "lz": 536.0,
        "lx": 67.0,
        "lz": 134.0,
        "initial_dt": 5e-4,
        "cfl_cutoff": 0.5,
        "Pr":7.0,
        "R0":1.1,
        "tau":1.0/3.0,
        "final_time": 100,
        "spatial_derivative_order": 2,
        "integrator_order": 2,
        "integrator": "explicit",
        "save_cadence": 0.5,
        # "load_from": 3,
        "dump_cadence": 0.5
    }

    I'm now using Stellmach et al 2011's definition of the Nusselt number, $Nu = 1-F_T$ where $F_T = <wT>$ is the flux averaged over the entire domain.

Best way to plot currently:
```
pipenv run python ../../plot.py --pretty --aspect_ratio 0.625 --figsize 10 16 --save --balance_cmap --smooth --ncores 4 tmp*.npy
```

Then render to video with
```
ffmpeg -r 30 -i tmp%04d.npy.png -c:v libx264 -vf "format=yuv420p,scale=1024:-2" out.mp4
```

## 2021-04-01

Finally managed to implement cosine and sine transforms manually using pre and post processing of the standard FFT.

## 2021-05-14

Useful snippet:

```fish
for f in *.png; convert $f -trim trim_$f; end
```
