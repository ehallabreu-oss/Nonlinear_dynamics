import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(4)

# n = 2
# semantics = np.random.randint(2, size=(n, n))

# cost = np.repeat(0, n).T
# prior = np.repeat(0.5, n)
# alpha = 4

# literal_listener = semantics*prior / np.sum(semantics*prior, axis=1, keepdims=True)

# utility = np.exp(alpha * (np.log(literal_listener) - cost)) # or alternatively: utility = literal_listener**alpha * np.exp(-cost)
# pragmatic_speaker = utility / np.sum(utility, axis=0)

# pragmatic_listener = pragmatic_speaker*prior / np.sum(pragmatic_speaker*prior, axis=1, keepdims=True)

# print(f'{semantics}\n\n{literal_listener}\n\n{pragmatic_speaker}\n\n{pragmatic_listener}')

# Simpler and smaller handpicked version for debugging
semantics = np.array([[0,1],
                      [1,1]])

cost = np.array([[0, 0]]).T
prior = np.array([[0.5, 0.5]])

sum_rows = np.sum(semantics, axis=1, keepdims=True)
literal_listener = np.divide(semantics, sum_rows) 

sum_columns = np.sum(literal_listener, axis=0)
pragmatic_speaker = np.divide(literal_listener, sum_columns)

sum_rows2 = np.sum(pragmatic_speaker, axis=1, keepdims=True)
pragmatic_listener = np.divide(pragmatic_speaker, sum_rows2)

print(f'{semantics}\n\n{sum_rows}\n\n{literal_listener}\n\n{sum_columns}\n\n{pragmatic_speaker}\n\n {sum_rows2}\n\n{pragmatic_listener}')

fig, ax = plt.subplots(2, 2)
ax[0,0].imshow(semantics, cmap='gray')
ax[0,0].set_title("Semantics")

ax[0,1].imshow(literal_listener, cmap='gray')
ax[0,1].set_title("Literal Listener")

ax[1,0].imshow(pragmatic_speaker, cmap='gray')
ax[1,0].set_title("Pragmatic Listener")

ax[1,1].imshow(pragmatic_listener, cmap='gray')
ax[1,1].set_title("Pragmatic Listener")

plt.tight_layout()
plt.show()