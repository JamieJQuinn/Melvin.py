import numpy as np
import matplotlib.pyplot as plt
import time
import sys

npy_fname = sys.argv[1]

png_fname = npy_fname[:-4] + "_xav.png"
data = np.load(npy_fname)
xav = np.mean(data, axis=0)
z = np.linspace(0, 536, xav.size)
density = z-xav
plt.plot(density, z)
plt.savefig(png_fname)
plt.close()
