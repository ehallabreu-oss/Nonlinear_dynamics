import numpy as np
import matplotlib.pyplot as plt

def logistic_map(n, x0, r):
    values = np.empty(n)
    values[0] = x0
    for i in range(1, n):
        values[i] = r * values[i-1] * (1 - values[i-1])
    return values

def bifurcation(n, x0, r_vec, k):
    x = np.zeros([len(r_vec), n-k])
    for i in range(len(r_vec)):
        x[i] = logistic_map(n, x0, r_vec[i])[k:]
    return x

step = 0.0005
r_vec = np.arange(2.4, 4+step, step)
xn = bifurcation(n=2000, x0=0.2, r_vec=r_vec, k=500)

R = np.repeat(r_vec, xn.shape[1])
X = xn.flatten()

plt.figure(figsize=(12, 7))
plt.scatter(R, X, color='black', s=0.001, alpha=0.1)
plt.title('Logistic map bifurcation diagram')
plt.xlabel('r')
plt.ylabel('x')
plt.xticks(np.arange(2.8, 3.6, 0.2))
plt.show()