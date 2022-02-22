import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import json


def ddz(var, dz):
    return (var[1:] - var[:-1]) / dz


def to_physical(var):
    return np.fft.irfft2(var) * var.shape[0] * var.shape[1]


npy_fname = sys.argv[1]
params_fname = sys.argv[2]


with open(params_fname, "r") as fp:
    params = json.load(fp)

R0 = params["R0"]

png_fname = npy_fname[:-4] + "_xav.png"
data = np.load(npy_fname)
tmp = to_physical(data["tmp"])
xi = to_physical(data["xi"])
# w = to_physical(data["w"])

# plt.imshow(tmp)
# plt.show()

# plt.imshow(xi)
# plt.show()

tmp_xav = np.mean(tmp, axis=0)
xi_xav = np.mean(xi, axis=0)

# plt.plot(tmp_xav-xi_xav)
# plt.show()

z = np.linspace(0, params["lz"], tmp_xav.size)
dz = z[1] - z[0]
R = np.zeros(tmp_xav.size)
R[:-1] = (1.0 + ddz(tmp_xav, dz)) / (1.0 / R0 + ddz(xi_xav, dz))
density = (1 - 1.0 / R0) * z - xi_xav + tmp_xav
# density =  xi_xav - tmp_xav
plt.plot(density, z)
plt.show()
# plt.savefig(png_fname)
plt.close()
