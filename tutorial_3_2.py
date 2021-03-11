import numpy as np
import matplotlib.pyplot as plt

tau = 0.1
T = 100
t = np.arange(0, T, tau)
n = t.shape[0]

C = 50
vr = -80
vt = -25
vpeak = 40
k = 1
a = 0.01
b = -20
c = -55
d = 150

g = np.zeros(n)
spike = np.zeros(n)
# spike[200:800:10] = 100
spike[200] = 100
psp_amp = 100
psp_decay = 10

v = np.zeros(n)
u = np.zeros(n)
v[0] = vr

for i in range(1, n):

    dvdt = (k * (v[i - 1] - vr) * (v[i - 1] - vt) - u[i - 1] + g[i-1]) / C
    dudt = a * (b * (v[i - 1] - vr) - u[i - 1])
    dgdt = (-g[i - 1] + psp_amp * spike[i - 1]) / psp_decay
    dt = t[i] - t[i - 1]

    v[i] = v[i - 1] + dvdt * dt
    u[i] = u[i - 1] + dudt * dt
    g[i] = g[i - 1] + dgdt * dt

    if v[i] >= vpeak:
        v[i - 1] = vpeak
        v[i] = c
        u[i] = u[i] + d

fig, ax = plt.subplots(2, 1, squeeze=False)
ax[0, 0].plot(t, v)
ax[1, 0].plot(t, g)
plt.show()
