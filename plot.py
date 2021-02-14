import numpy as np
import matplotlib.pyplot as plt
import time
import sys

npy_fname = sys.argv[1]

extent = [0, 335, 0 ,536]

png_fname = npy_fname[:-3] + "png"
start = time.time()
data = np.load(npy_fname)
vmax = max(abs(data.min()), data.max())
plt.imshow(data.T, origin='lower', extent=extent, cmap='RdBu', vmax=vmax, vmin=-vmax)
plt.savefig(png_fname)
diff = time.time() - start
print(diff)
