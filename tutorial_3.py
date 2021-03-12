import numpy as np
import matplotlib.pyplot as plt

tau = 0.1
T = 10
A = 1
t_peak = 1
t = np.arange(0, T, tau)
n = t.shape[0]
v_psp = A * t * np.exp(-t/t_peak)

fig, ax = plt.subplots(1, 1, squeeze=False)
ax[0, 0].plot(t, v_psp)
ax[0, 0].set_xlabel('x')
ax[0, 0].set_ylabel('V')
plt.show()




v_psp = np.zeros(n)
g = np.zeros(n)
spike = np.zeros(n)
spike[20] = 1

for i in range(n):
  dgdt = (-g[i-1] + A * spike[i-1]) / t_peak
  dt = t[i] - t[i-1]

  g[i] = g[i-1] + dgdt * dt

fig, ax = plt.subplots(1, 1, squeeze=False)
ax[0,0].plot(t, g)
ax[0,0].set_xlabel('x')
ax[0,0].set_ylabel('g')
plt.show()




tau = 0.1
T = 100
A = 1
t_peak = 2
t = np.arange(0, T, tau)
n = t.shape[0]

v_psp = np.zeros(n)
g = np.zeros(n)
spike = np.zeros(n)
# spike[20:100:20] = 1
spike[20] = 1
E = -10
g_leak = 1
C=1
for i in range(n):
  dvdt = (g[i-1]*(v_psp[i-1]- E) - g_leak * v_psp[i-1]) / C
  dgdt = (-g[i-1] + A * spike[i-1]) / t_peak
  dt = t[i] - t[i-1]

  v_psp[i] = v_psp[i-1] + dvdt * dt
  g[i] = g[i-1] + dgdt * dt

fig, ax = plt.subplots(2, 1, squeeze=False)

ax[0,0].plot(t, v_psp)
ax[0,0].set_xlabel('t')
ax[0,0].set_ylabel('V')

ax[1,0].plot(t, g)
ax[1,0].set_xlabel('t')
ax[1,0].set_ylabel('g')

plt.show()



