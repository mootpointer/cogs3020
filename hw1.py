import numpy as np
import matplotlib.pyplot as plt

tau = 0.01  # how many ms one computer step represents
T = 1000  # total ms of the simulation
t = np.arange(0, T, tau)  # initialise time array

h = np.zeros(t.shape)
m = np.zeros(t.shape)
n = np.zeros(t.shape)

v = np.zeros(t.shape)  # initialise potential with zeros
vr = -65  # define initial membrane potential
v[0] = vr  # set initial membrane potential

I = [0] * (t.shape[0] // 3) + [1] * (t.shape[0] // 3) + [0] * (t.shape[0] // 3)
# I = [0] * t.shape[0]
C = 50

g_na = 120
g_k = 36
g_leak = 0.3

E_na = 115 + vr
E_k = -6 + vr
E_leak = 10.6 + vr + 7 * 0


def alpha_func_h(v):
    y = 0.07 * np.exp((vr - v) / 20)
    return y


def alpha_func_m(v):
    y = (2.5 - 0.1 * (v - vr)) / (np.exp(2.5 - 0.1 * (v - vr)) - 1)
    return y


def alpha_func_n(v):
    y = (0.1 - 0.01 * (v - vr)) / (np.exp(1.0 - 0.1 * (v - vr)) - 1)
    return y


def beta_func_h(v):
    y = 1 / (1 + np.exp(3 - 0.1 * (v - vr)))
    return y


def beta_func_m(v):
    y = 4 * np.exp((vr - v) / 18)
    return y


def beta_func_n(v):
    y = 0.125 * np.exp((vr - v) / 80)
    return y


h[0] = alpha_func_h(vr) / (alpha_func_h(vr) + beta_func_h(vr))
m[0] = alpha_func_m(vr) / (alpha_func_m(vr) + beta_func_m(vr))
n[0] = alpha_func_n(vr) / (alpha_func_n(vr) + beta_func_n(vr))

for i in range(1, t.shape[0]):

    I_na = g_na * h[i - 1] * m[i - 1]**3 * (v[i - 1] - E_na)
    I_k = g_k * n[i - 1]**4 * (v[i - 1] - E_k)
    I_leak = g_leak * (v[i - 1] - E_leak)

    dvdt = I[i - 1] - (I_na + I_k + I_leak)

    dhdt = alpha_func_h(v[i - 1]) * (1 - h[i - 1]) - beta_func_h(
        v[i - 1]) * h[i - 1]
    dmdt = alpha_func_m(v[i - 1]) * (1 - m[i - 1]) - beta_func_m(
        v[i - 1]) * m[i - 1]
    dndt = alpha_func_n(v[i - 1]) * (1 - n[i - 1]) - beta_func_n(
        v[i - 1]) * n[i - 1]

    # delta t
    dt = t[i] - t[i - 1]

    # Euler's update
    v[i] = v[i - 1] + dvdt * dt
    h[i] = h[i - 1] + dhdt * dt
    m[i] = m[i - 1] + dmdt * dt
    n[i] = n[i - 1] + dndt * dt

fig, ax, = plt.subplots(2, 1, squeeze=False)
ax[0, 0].plot(t, v)
ax[0, 0].set_ylabel('v')
ax[0, 0].set_xlabel('t')

ax[1, 0].plot(t, n**4, label='K')
ax[1, 0].plot(t, h * m**3, label='Na')
ax[1, 0].set_ylabel('n')
ax[1, 0].set_xlabel('t')
plt.legend()
plt.show()
