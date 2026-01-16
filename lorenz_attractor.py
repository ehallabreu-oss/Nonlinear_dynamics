import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def lorenz_solver_rk4(x0, y0, z0, a, b, c, length, dt):
    trajectory = np.empty([length, 3])
    trajectory[0] = np.array([x0, y0, z0])

    def slope(state):
        x, y, z = state
        derivative_vec = np.array([a*(y-x), x*(b-z)-y, x*y - c*z])
        return derivative_vec
    
    for n in range(1, length):
        state = trajectory[n-1]

        k1 = slope(state)
        k2 = slope(state + dt/2 * k1)
        k3 = slope(state + dt/2 * k2)
        k4 = slope(state + dt * k3)

        trajectory[n] = state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    return trajectory

lorenz = lorenz_solver_rk4(x0=1, y0=1, z0=1, a=10, b=28, c=8/3, length=50000, dt=0.01)

x, y, z = lorenz[:,0], lorenz[:,1], lorenz[:,2] 

# rgb = rgb = np.column_stack([
#     (x - x.min()) / (x.max() - x.min()),
#     (y - y.min()) / (y.max() - y.min()),
#     (z - z.min()) / (z.max() - z.min())
# ])

plt.style.use('dark_background')
ax = plt.figure(figsize=(8,8)).add_subplot(projection='3d')

ax.plot(x,y,z, color='white', lw=0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Lorenz attractor')

plt.show()

