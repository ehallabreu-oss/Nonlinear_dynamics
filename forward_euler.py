import numpy as np
import matplotlib.pyplot as plt

# forward euler for simple harmonic oscilator

def forward_euler(x0, v0, length, dt, k, m, g):
    trajectory = np.empty([length, 2])
    print(trajectory.shape)
    trajectory[0] = np.array([x0, v0])
    print(trajectory[0].shape)


    for n in range(1, length):
        trajectory[n] = np.array([
            trajectory[n-1, 0] + dt * trajectory[n-1, 1],   # = x[n]
            trajectory[n-1, 1] + dt * ((-k/m) * trajectory[n-1, 0] + g) # = v[n]
        ])

    return trajectory

trajectory = forward_euler(x0=0.5, v0=0.5, length=6, dt=0.1, k=2, m=0.5, g=0)
print(trajectory)

plt.plot(trajectory[:,0], trajectory[:,1])
plt.quiver
plt.show()


