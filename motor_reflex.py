import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

def rk4(f, x, t, dt):
    k1 = f(x)
    k2 = f(x + dt*k1/2)
    k3 = f(x + dt*k2/2)
    k4 = f(x + dt*k3)
    
    return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)


def get_rk4(f, dt):
    def rk4_scan_fn(carry, t):
        next_state = rk4(f, carry, t, dt)
        return next_state, next_state
    
    return rk4_scan_fn

def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

def motor_reflex(X, w1 = 1., w2 = 1., w3 = 0.2):   
    x1, x2, x3 = X 
    return jnp.array([w1*x2 - x3*x1,
                      -w2*x1,
                      -w3*x3 + w3
                      ])

inital_condition = jnp.array([5., 5., 0.])
dur = 80
dt = 0.01

_, states = jax.lax.scan(get_rk4(motor_reflex, dt), inital_condition, length=dur/dt)

solution = states.T

ax = plt.figure().add_subplot(projection='3d')
ax.scatter(inital_condition[0], inital_condition[1], inital_condition[2])
ax.plot(solution[0], solution[1], solution[2], lw=1)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('I')

plt.show()

time = jnp.arange(0, dur, dt)

plt.figure()
plt.xlabel('time')
plt.ylabel('Inhibition')
plt.plot(time, solution[2])
plt.show()

