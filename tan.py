import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# The Izhikevich model -- TAN
tau = 1
C = 100.0
vr = -75.0
vt = -45.0
k = 1.2
a = 0.01
b = 5.0
c = -56.0
d = 130.0
vpeak = 60.0
E_base = 950.0

n_steps = 900

input_onset = 200
input_offset = 300
input_amp = 2000
mod_amp = 2000
E_decay = 0.99

E = np.zeros(n_steps)
E_mod = np.zeros(n_steps)

for i in range(n_steps):
    if i >= input_onset and i < input_offset:
        E[i] = input_amp
        E_mod[i] = mod_amp

    if i >= input_offset:
        E_mod[i] = E_decay * E_mod[i - 1]

v = np.zeros(n_steps)
u = np.zeros(n_steps)
o = np.zeros(n_steps)

v[0] = vr

alpha = 0.01
beta = 1.0
o = np.zeros(n_steps)

for i in range(n_steps - 1):
    v[i + 1] = v[i] + tau * (k * (v[i] - vr) *
                             (v[i] - vt) - u[i] + E_base + E[i]) / C
    u[i + 1] = u[i] + tau * a * (b * (v[i] - vr) - u[i] + E_mod[i])
    o[i + 1] = o[i] + alpha * (np.heaviside(v[i] - vt, vt) - o[i])

    if v[i + 1] >= vpeak:
        v[i] = vpeak
        v[i + 1] = c
        u[i + 1] = u[i + 1] + d

fig, ax = plt.subplots(nrows=2, ncols=1)
ax[0].plot(np.arange(0, n_steps, 1), E)
ax[0].plot(np.arange(0, n_steps, 1), E_mod)
ax[0].set_ylabel('Input')

ax[1].plot(np.arange(0, n_steps, 1), v, 'C0')
ax[1].set_ylabel('Membrane V')

ax1 = ax[1].twinx()
ax1.plot(np.arange(0, n_steps, 1), o, 'C1')
ax1.set_ylabel('Output')
ax1.yaxis.label.set_color('C1')

plt.tight_layout()
plt.show()
