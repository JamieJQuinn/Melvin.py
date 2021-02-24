import sys
import numpy as np
import matplotlib.pyplot as plt

window_width = 2

filenames = sys.argv[1:]
print(f"filenames: {filenames}")

max_t = 0

for f in filenames:
    dataset = np.load(f)
    t = dataset['t']
    ke = dataset['values']
    if ke.dtype == 'complex128':
        ke = np.imag(ke)
    cumsum_vec = np.cumsum(np.insert(ke, 0, 0)) 
    ke_smooth = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    plt.plot(t[:-window_width+1], ke_smooth, label=f)
    max_t = max(max_t, t[-1])

# plt.ylim(0, 1000)
plt.xlim(0, max_t)
plt.legend()
plt.show()
