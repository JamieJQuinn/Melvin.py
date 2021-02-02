import numpy as np
import time
from utility import sech
import matplotlib.pyplot as plt

from Parameters import Parameters
from SpectralTransformer import SpectralTransformer
from Variable import Variable
from TimeDerivative import TimeDerivative
from LaplacianSolver import LaplacianSolver
from Integrator import Integrator

MODULE=np

def plot_physical(pvar):
    plt.imshow(pvar.T)
    plt.show()

def load_initial_conditions(params, w):
    x = np.linspace(0, params.lx, params.nx, endpoint = False)
    z = np.linspace(0, params.lz, params.nz, endpoint = False)
    X, Z = np.meshgrid(x, z, indexing='ij')

    epsilon = 1e-2
    sigma = 0.2
    h = 0.05
    pert_n = 1

    ## Set vorticity
    w0_p = np.power(sech((Z-0.5)/h), 2)/h
    w0_pert_p = epsilon * params.kn*np.cos(pert_n*params.kn*X+np.pi)\
    *(np.exp(-((Z-0.5)/sigma)**2))
    w0_p += w0_pert_p

    w.set_physical(w0_p)
    w.to_spectral()

    ## Set ink
    # c0_p = np.sin(1*params.km*Z)

    # c.set_physical(c0_p)
    # c.to_spectral()

def calc_nl(var, vx, vz, st):
    result = vx*var.ddx() + vz*var.ddz()
    return st.to_spectral(result)

def main():
    PARAMS = {
        "nx": 4**3,
        "nz": 2**6,
        "lx": 1.0,
        "lz": 1.0,
        "dt": 1e-4,
        "Re": 1e5,
        "integrator_order": 4
    }
    params = Parameters(PARAMS)
    st = SpectralTransformer(params, MODULE)
    lap_solver = LaplacianSolver(params, MODULE)
    integrator = Integrator(params)

    # Create mode number matrix
    n = np.concatenate((np.arange(0, params.nn+1),  np.arange(-params.nn, 0)))
    m = np.arange(0, params.nm)
    n, m = np.meshgrid(n, m, indexing='ij')

    w = Variable(params, MODULE, st)
    dw = TimeDerivative(params, MODULE)

    psi = Variable(params, MODULE, st)
    ux = Variable(params, MODULE, st)
    uz = Variable(params, MODULE, st)

    load_initial_conditions(params, w)

    T = 1.0
    t = 0.0
    print_time = 0.1
    print_track = 0.0

    ke = 0.0

    start = time.time()
    while t < T:
        if print_track < t:
            print_track += print_time
            print(ke)
        lap_solver.solve(w.gets(), psi.gets())
        psi._sdata[0,0] = 0.0

        ux[:] = 1j*params.km*m*psi[:]
        ux.to_physical()
        uz[:] = -1j*params.kn*n*psi[:]
        uz.to_physical()

        ke = 0.5*MODULE.sum(ux.getp()**2 + uz.getp()**2)/(params.nx*params.nz)

        # Predictor
        w.to_physical()
        dw[:] = -calc_nl(w, ux.getp(), uz.getp(), st) + 1.0/params.Re*lap_solver.lap*w[:]
        w[:] += integrator.predictor(dw)

        dw.advance()

        ## Corrector
        w.to_physical()
        dw[:] = -calc_nl(w, ux.getp(), uz.getp(), st) + 1.0/params.Re*lap_solver.lap*w[:]
        w[:] += integrator.corrector(dw)

        t += params.dt

    end = time.time() - start
    print(end)


main()
