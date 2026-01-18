import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# loading time series (driven pendulum)
amplitude = np.loadtxt('amplitude.dat')
time = np.arange(0, len(amplitude))

# visualising time series
plt.plot(time[:2000], amplitude[:2000])
plt.show()

def delay_coordinate(time_series, delay, dim):
    embedding = np.empty([dim, len(time_series) - delay*(dim-1)])

    for n in range(dim):
        if n == 0:
            embedding[n] = time_series[delay * (dim-1):]
        else:
            embedding[n] = time_series[delay * (dim-1-n) : -(delay * n)]

    return embedding

# adjustable params
dimension = 3
init_delay = 8
low_bound = 0
up_bound = 350

# fig and manipulated line
fig, ax = plt.subplots()
line, = ax.plot(delay_coordinate(amplitude, init_delay, dimension)[0], 
                delay_coordinate(amplitude, init_delay, dimension)[1],
                lw=0.3)
ax.set_xlabel('x(t)')
ax.set_ylabel('x(t - 2τ)')

fig.subplots_adjust(left=0.25, bottom=0.25)

# make horizontal slider
tau = fig.add_axes([0.25, 0.1, 0.65, 0.03])
step_delay = np.arange(low_bound, up_bound)

tau_slider = Slider(
    ax = tau,
    label = 'Delay (n samples)',
    valmin = low_bound,
    valmax = up_bound,
    valinit = init_delay,
    valstep = step_delay 
)

def update(val):
    delay = tau_slider.val
    line.set_xdata(delay_coordinate(amplitude, delay, dimension)[0])
    line.set_ydata(delay_coordinate(amplitude, delay, dimension)[1])
    fig.canvas.draw_idle()

tau_slider.on_changed(update)

plt.show()

#without slider
embedded = delay_coordinate(time_series=amplitude, delay=8, dim=3)
x1, x2, x3 = embedded[0], embedded[1], embedded[2]

# plt.plot(x1, x3, lw=0.3)
# plt.title('Time series theta')
# plt.xlabel('x(t)')
# plt.ylabel('x(t - 2τ)')

#3d plot
ax = plt.figure(figsize=(8,8)).add_subplot(projection='3d')
ax.plot(x1, x2, x3, lw=0.3)
ax.set_xlabel('x(t)')
ax.set_ylabel('x(t - τ)')
ax.set_zlabel('x(t - 2τ)')
plt.show()