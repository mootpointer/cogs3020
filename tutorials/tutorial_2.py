# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
T = 100
tau = 0.01
t = np.arange(0, T, tau)
n = t.shape[0]

v = np.zeros(n)

b = np.concatenate(([0.0] * (n // 3),
                    [1.001] * (n // 3),
                    [0.0] * (n // 3)
                    ))


for i in range(1, n):
    dvdt = b[i - 1] - v[i - 1]
    dt = t[i] - t[i - 1]

    v[i] = v[i-1] + dvdt * dt

    if v[i] > 1:
        v[i] = 0

fig, ax, = plt.subplots(1, 1, squeeze=False)
ax[0, 0].plot(t, v)
ax[0, 0].set_ylabel('v')
ax[0, 0].set_xlabel('t')
plt.show()
