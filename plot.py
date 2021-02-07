import numpy as np
import matplotlib.pyplot as plt
import time

def form_fname(name, idx):
    return name + '{:04d}'.format(idx) + ".npy"

for idx in range(300):
    for name in ['tmp', 'w', 'xi']:
    # for name in ['tmp']:
        fname = form_fname(name, idx)
        start = time.time()
        data = np.load(fname)
        plt.imsave(fname+".png", data.T, cmap='RdBu')
        # plt.imsave(fname+".png", data.T)
        plt.close()
        diff = time.time() - start
        print(idx, diff)
