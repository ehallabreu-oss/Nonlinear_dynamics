import numpy as np
import matplotlib.pyplot as plt

# load lorenz attarctor data
lorenz_full = np.genfromtxt('CapDimData.dat', dtype=float, delimiter=',') # (14000, 3)
lorenz = lorenz_full[:8000]
# xz projection
X, Z = lorenz[:,0], lorenz[:,2]
coords = np.stack([X, Z])

res = 0.3
hindices = np.arange(np.min(X), np.max(X)+res, res) 
vindices = np.arange(np.min(Z), np.max(Z)+res, res) 

grid = np.zeros([len(vindices), len(hindices)])

for n in range(coords.shape[1]):
    x = coords[0,n]
    z = coords[1,n]
    i = int(np.ceil((z - np.min(Z)) / res))
    j = int(np.ceil((x - np.min(X)) / res))
    grid[i,j] = 1

plt.matshow(grid)
plt.show()