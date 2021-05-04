import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()


def animate(i):
    ke = np.load("kinetic_energy.npy")
    ax.clear()
    ax.plot(ke[0], ke[1])


ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()
