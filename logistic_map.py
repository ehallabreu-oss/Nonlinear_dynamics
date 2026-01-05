import numpy as np
import matplotlib.pyplot as plt

def logistic_map(n, x0, r):
    values = np.empty(n)
    values[0] = x0
    for i in range(1, n):
        values[i] = r * values[i-1] * (1 - values[i-1])
    return values

n = 200
xn = logistic_map(n, x0=0.2, r=2)
xhat = logistic_map(n, x0=0.200001, r=2)
diff = np.abs(xn - xhat)

xn_1 = logistic_map(n, x0=0.2, r=3.4)
xhat_1 = logistic_map(n, x0=0.200001, r=3.4)
diff_1 = np.abs(xn_1 - xhat_1)

xn_2 = logistic_map(n, x0=0.2, r=3.72)
xhat_2 = logistic_map(n, x0=0.200001, r=3.72)
diff_2 = np.abs(xn_2 - xhat_2)

avg_diff = np.sum(diff_2)/n

iterations = np.arange(n)

plt.scatter(iterations, diff_2, color='blue', s=5)
plt.plot(iterations, diff_2, color='gray', alpha=0.5, linewidth=0.2)
plt.xlabel('n')
plt.xticks(np.linspace(0, 200, 5))
plt.ylabel('|xn - xhat|')
plt.show()