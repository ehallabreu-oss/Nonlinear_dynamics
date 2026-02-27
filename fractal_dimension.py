import numpy as np
import matplotlib.pyplot as plt

# load lorenz attarctor data
lorenz = np.genfromtxt('CapDimData.dat', dtype=float, delimiter=',') # (14000, 3)
# xz projection
X, Z = lorenz[:,0], lorenz[:,2]
coords = np.stack([X, Z])

def box_counter(coords, epsilon):
    
    hindices = np.arange(np.min(X), np.max(X)+epsilon, epsilon) 
    vindices = np.arange(np.min(Z), np.max(Z)+epsilon, epsilon) 
    
    grid = np.zeros([len(vindices), len(hindices)])

    for n in range(coords.shape[1]):
        x = coords[0,n]
        z = coords[1,n]
        i = int(np.ceil((z - np.min(Z)) / epsilon))
        j = int(np.ceil((x - np.min(X)) / epsilon))
        grid[i,j] = 1

    return grid

size = 40
epsilon_range = np.linspace(0.5, 1, size)
box_number = np.empty(size)

for n in range(size):
    epsilon = epsilon_range[n]
    grid = box_counter(coords, epsilon)
    box_number[n] = np.sum(grid)


line = np.stack([-np.log(epsilon_range), np.log(box_number)])

plt.scatter(line[0], line[1])
plt.xlabel('log(1/ε)')
plt.ylabel('log(N(ε))')
plt.title('Capacity dimension')
plt.show()

# fitting a line to find the slope
slope_init = 1
bias_init = 1
learning_rate = 0.2

def linear_regression(data, slope, bias, alpha):
    x, y = data[0], data[1]

    for epoch in range(500):
        predicted = slope * x + bias
        
        cost = np.sum((predicted - y)**2)/(2*len(x))

        if epoch // 10 == epoch / 10:
            print(cost)
        
        slope -= (alpha/len(x)) * np.sum((predicted - y) * x)
        bias -= (alpha/len(x)) * np.sum(predicted - y) 

    return slope, bias, predicted

slope, intercept, predicted  = linear_regression(line, slope_init, bias_init, learning_rate)

print(f'slope (and hence capacitty dimension) is {slope} \n intercept is {intercept}')

plt.scatter(line[0], line[1], color='blue', label='real')
plt.plot(line[0], predicted, color='red', label='fitted')
plt.show()
