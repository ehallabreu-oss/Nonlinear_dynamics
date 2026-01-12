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


def backward_euler(x0, v0, length, dt, k, m, g):
    forward = np.empty([length, 2]) 
    forward[0] = np.array([x0, v0]) 
    
    backward = np.empty([length, 2])
    backward[0] = np.array([x0, v0]) 

    for n in range(1, length):
        forward[n] = np.array([
            backward[n-1, 0] + dt * forward[n-1, 1],               # = xFE[n]
            backward[n-1, 1] + dt * ((-k/m) * forward[n-1, 0] + g) # = vFE[n]
        ])
        
        for n in range(1, length):
            backward[n] = np.array([
                backward[n-1, 0] + dt * forward[n, 1],               # = x[n]
                backward[n-1, 1] + dt * ((-k/m) * forward[n, 0] + g) # = v[n]
            ])

    return backward

def trapezoidal(x0, v0, length, dt, k, m, g):
    forward = np.empty([length, 2]) 
    forward[0] = np.array([x0, v0]) 
    
    backward = np.empty_like(forward)
    backward[0] = np.array([x0, v0]) 

    average = np.empty_like(forward)
    average[0] = np.array([x0, v0]) 


    for n in range(1, length):
        forward[n] = np.array([
            average[n-1, 0] + dt * forward[n-1, 1],   # = xFE[n]
            average[n-1, 1] + dt * ((-k/m) * forward[n-1, 0] + g) # = vFE[n]
        ])
        
        for n in range(1, length):
            backward[n] = np.array([
                backward[n-1, 0] + dt * forward[n, 1],  # = x[n]
                backward[n-1, 1] + dt * ((-k/m) * forward[n, 0] + g) # = v[n]
            ])

            for n in range(1, length):
                average[n] = average[n-1] + ((forward[n-1] + backward[n-1]) / 2)

    return average
    
# trying the implicit way
def implicit_euler(x0, v0, length, dt, k, m, g):
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

trajectory = forward_euler(x0=-1, v0=-2, length=500, dt=0.1, k=2, m=0.5, g=0)
print(trajectory[:6])

trajectory_backward = backward_euler(x0=-1, v0=-2, length=500, dt=0.1, k=2, m=0.5, g=0)
print(trajectory_backward[:6])

trapezoid = trapezoidal(x0=-1, v0=-2, length=500, dt=0.1, k=2, m=0.5, g=0)

# plt.plot(trajectory_backward[:,0], trajectory_backward[:,1], color='blue', label='backward')
# plt.plot(trajectory[:,0], trajectory[:,1], color='red', label='forward')
plt.plot(trapezoid[:,0], trapezoid[:,1], color='green', label='trapezoid')
plt.title('harmonic oscilator euler solvers')
plt.legend()
plt.xlabel('x')
plt.ylabel('v')
plt.grid()
plt.show()