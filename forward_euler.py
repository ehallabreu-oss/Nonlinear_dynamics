import numpy as np
import matplotlib.pyplot as plt

# forward euler for simple harmonic oscilator

def forward_euler(x0, v0, length, dt, k, m, g):
    
    trajectory = np.empty([length, 2])
    trajectory[0] = np.array([x0, v0])

    for n in range(1, length):
        trajectory[n] = np.array([
            trajectory[n-1, 0] + dt * trajectory[n-1, 1],   # = x[n]
            trajectory[n-1, 1] + dt * ((-k/m) * trajectory[n-1, 0] + g) # = v[n]
        ])

    return trajectory

trajectory = forward_euler(x0=-1, v0=-2, length=50, dt=0.1, k=2, m=0.5, g=0)
print(trajectory)

def backward_euler(x0, v0, length, dt, k, m, g):
    trajectory = np.empty([length, 2])
    trajectory[0] = np.array([x0, v0])

    A = np.array([
        [1, -dt],
        [dt*(k/m), 1]
    ])

    for n in range(1, length):
        b = np.array([
            trajectory[n-1, 0],
            trajectory[n-1, 1] + dt * g
        ])
        trajectory[n] = np.linalg.solve(A, b)

    return trajectory

trajectory_backward = backward_euler(x0=-1, v0=-2, length=50, dt=0.1, k=2, m=0.5, g=0)
# print(trajectory_backward)

plt.plot(trajectory_backward[:,0], trajectory_backward[:,1], color='blue', label='backward')
plt.plot(trajectory[:,0], trajectory[:,1], color='red', label='forward')
plt.title('harmonic oscilator euler solver')
plt.legend()
plt.xlabel('x')
plt.ylabel('v')
plt.grid()
plt.show()