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
    
    backward = np.empty_like(forward)
    backward[0] = np.array([x0, v0]) 

    for n in range(1, length):
        forward[n] = np.array([
            backward[n-1, 0] + dt * forward[n-1, 1],               # = xFE[n]
            backward[n-1, 1] + dt * ((-k/m) * forward[n-1, 0] + g) # = vFE[n]
        ])
        
        backward[n] = np.array([
            backward[n-1, 0] + dt * forward[n, 1],               # = x[n]
            backward[n-1, 1] + dt * ((-k/m) * forward[n, 0] + g) # = v[n]
        ])

    return backward

def trapezoidal_first(x0, v0, length, dt, k, m, g):

    forward = np.empty([length, 2]) 
    forward[0] = np.array([x0, v0]) 

    average = np.empty_like(forward)
    average[0] = np.array([x0, v0]) 

    for n in range(1, length):
        forward[n] = np.array([
            average[n-1, 0] + dt * forward[n-1, 1],               # = xFE[n]
            average[n-1, 1] + dt * ((-k/m) * forward[n-1, 0] + g) # = vFE[n]
        ])
        
        average[n] = np.array([
            average[n-1, 0] + dt/2 * (forward[n, 1] + forward[n-1, 1]),               
            average[n-1, 1] + dt/2 * (((-k/m) * forward[n, 0] + g) + ((-k/m) * forward[n-1, 0] + g))
        ])
                    
    return average

def trapezoidal(x0, v0, length, dt, k, m, g):
    trajectory = np.empty([length, 2])
    trajectory[0] = np.array([x0, v0])

    def slope(state):
        x, v = state
        return np.array([v, (-k/m)*x + g])
    
    for n in range(1, length):
        state = trajectory[n-1]

        forward = slope(state)
        backward = slope(state + dt*forward)

        trajectory[n] = state + dt/2 * (forward + backward)

    return trajectory


def RK4(x0, v0, length, dt, k, m, g):
    trajectory = np.empty([length, 2])
    trajectory[0] = np.array([x0, v0])

    def slope(state):
        x, v = state    # a coordinate/point
        derivative_vec = np.array([v, (-k/m) * x + g]) 
        return derivative_vec # a vector/derivate at that point
        
    for n in range(1, length):
        state = trajectory[n-1]

        k1 = slope(state)   # slope/derivative/vector at the current point
        k2 = slope(state + dt/2 * k1)  # derivative at the point k1 brings us to / tip of the k1 vec (not realy because it's scaled)
        k3 = slope(state + dt/2 * k2)  # derivative at the point k2 brings us to (from the same state!)
        k4 = slope(state + dt * k3)    # derivative at the point k3 brings us to (again from the same state!)

        trajectory[n] = state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    return trajectory

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

# params
x0 = -5
v0 = 0
spring_force = 10
mass = 7
gravity = 0

length = 200
dt = 0.1

trajectory = forward_euler(x0, v0, length, dt, spring_force, mass, gravity)
trajectory_backward = backward_euler(x0, v0, length, dt, spring_force, mass, gravity)
trapezoid = trapezoidal(x0, v0, length, dt, spring_force, mass, gravity)
runge_kutta = RK4(x0, v0, length, dt, spring_force, mass, gravity)

# plt.plot(trajectory[:,0], trajectory[:,1], color='red', label='forward')
plt.plot(trajectory_backward[:,0], trajectory_backward[:,1], color='blue', label='backward')
plt.plot(trapezoid[:,0], trapezoid[:,1], color='green', label='trapezoid')
plt.plot(runge_kutta[:,0], runge_kutta[:,1], color='orange', label='RK4')
plt.title('SHO euler solvers')
plt.legend()
plt.xlabel('x')
plt.ylabel('v')
plt.grid()
plt.show()