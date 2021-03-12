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

psp_amp = 1e5
psp_decay = 10

g = np.zeros(n)
spike = np.zeros(n)
spike[200:800:20] = 1

v1 = np.zeros(n)
u1 = np.zeros(n)
g1 = np.zeros(n)
spike1 = np.zeros(n)
v1[0] = vr

v2 = np.zeros(n)
u2 = np.zeros(n)
g2 = np.zeros(n)
spike2 = np.zeros(n)
v2[0] = vr

v3 = np.zeros(n)
u3 = np.zeros(n)
g3 = np.zeros(n)
spike3 = np.zeros(n)
v3[0] = vr

w_01 = 1.0
w_12 = 0.5
w_23 = 0.1

for i in range(1, n):

    dt = t[i] - t[i - 1]

    # external input
    dgdt = (-g[i - 1] + psp_amp * spike[i - 1]) / psp_decay
    g[i] = g[i - 1] + dgdt * dt

    # neuron 1
    dvdt1 = (k * (v1[i - 1] - vr) * (v1[i - 1] - vt) - u1[i - 1] + w_01 * g[i-1]) / C
    dudt1 = a * (b * (v1[i - 1] - vr) - u1[i - 1])
    dgdt1 = (-g1[i - 1] + psp_amp * spike1[i - 1]) / psp_decay
    v1[i] = v1[i - 1] + dvdt1 * dt
    u1[i] = u1[i - 1] + dudt1 * dt
    g1[i] = g1[i - 1] + dgdt1 * dt
    if v1[i] >= vpeak:
        v1[i - 1] = vpeak
        v1[i] = c
        u1[i] = u1[i] + d
        spike1[i] = 1

    # neuron 2
    dvdt2 = (k * (v2[i - 1] - vr) * (v2[i - 1] - vt) - u2[i - 1] + w_12 * g1[i-1]) / C
    dudt2 = a * (b * (v2[i - 1] - vr) - u2[i - 1])
    dgdt2 = (-g2[i - 1] + psp_amp * spike2[i - 1]) / psp_decay
    v2[i] = v2[i - 1] + dvdt2 * dt
    u2[i] = u2[i - 1] + dudt2 * dt
    g2[i] = g2[i - 1] + dgdt2 * dt
    if v2[i] >= vpeak:
        v2[i - 1] = vpeak
        v2[i] = c
        u2[i] = u2[i] + d
        spike2[i] = 1

    # neuron 3
    dvdt3 = (k * (v3[i - 1] - vr) * (v3[i - 1] - vt) - u3[i - 1] + w_23 * g2[i-1]) / C
    dudt3 = a * (b * (v3[i - 1] - vr) - u3[i - 1])
    dgdt3 = (-g3[i - 1] + psp_amp * spike3[i - 1]) / psp_decay
    v3[i] = v3[i - 1] + dvdt3 * dt
    u3[i] = u3[i - 1] + dudt3 * dt
    g3[i] = g3[i - 1] + dgdt3 * dt
    if v3[i] >= vpeak:
        v3[i - 1] = vpeak
        v3[i] = c
        u3[i] = u3[i] + d
        spike3[i] = 1



fig, ax = plt.subplots(2, 3, squeeze=False)
ax[0, 0].plot(t, g)
ax[1, 0].plot(t, v1)
ax[0, 1].plot(t, g1)
ax[1, 1].plot(t, v2)
ax[0, 2].plot(t, g2)
ax[1, 2].plot(t, v3)
plt.show()
