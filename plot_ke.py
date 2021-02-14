import numpy as np
import matplotlib.pyplot as plt

ke = np.load("kinetic_energy.npy")
# plt.semilogy(ke[0], ke[1])
plt.plot(ke[0], ke[1])
plt.show()
