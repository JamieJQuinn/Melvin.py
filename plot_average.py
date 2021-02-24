import numpy as np
import matplotlib.pyplot as plt
import time
import sys

def ddz(var, dz):
    return (var[1:] - var[:-1])/dz

def to_physical(var):
    return np.fft.irfft2(var) * 2**19

R0 = 1.1

npy_fname = sys.argv[1]

png_fname = npy_fname[:-4] + "_xav.png"
data = np.load(npy_fname)
tmp = to_physical(data['tmp'])
xi = to_physical(data['xi'])
w = to_physical(data['w'])

tmp_xav = np.mean(tmp, axis=0)
xi_xav = np.mean(xi, axis=0)
z = np.linspace(0, 536, tmp_xav.size)
dz = z[1] - z[0]
Ft = 
R = np.zeros(tmp_xav.size)
R[:-1] = (1.0 + ddz(tmp_xav, dz))/(1.0/R0 + ddz(xi_xav, dz))
density = (1-1.0/R)*z - xi_xav + tmp_xav
# density =  xi_xav - tmp_xav
plt.plot(density, z)
plt.show()
# plt.savefig(png_fname)
plt.close()
