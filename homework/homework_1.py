# %% Import our libraries
import numpy as np
import matplotlib.pyplot as plt

# %% Define our hh model


def hh_model(tau, T, inj=None, E_leak=10.6):
    t = np.arange(0, T, tau)  # initialise time array

    h = np.zeros(t.shape)
    m = np.zeros(t.shape)
    n = np.zeros(t.shape)

    v = np.zeros(t.shape)  # initialise potential with zeros
    vr = -65  # define initial membrane potential
    v[0] = vr  # set initial membrane potential

    I = [4] * t.shape[0]
    C = 50

    g_na = 120
    g_k = 36
    g_leak = 0.3

    E_na = 115 + vr
    E_k = -6 + vr
    E_leak = E_leak + vr

    if inj:
        I_inj = inj(t)
    else:
        I_inj = np.zeros(t.shape)

    def alpha_func_h(v): return 0.07 * np.exp((vr - v)/20)
    def alpha_func_m(v): return (2.5-0.1*(v-vr)) / (np.exp(2.5-0.1*(v-vr))-1)
    def alpha_func_n(v): return (0.1-0.01*(v-vr)) / (np.exp(1.0-0.1*(v-vr))-1)

    def beta_func_h(v): return 1/(1+np.exp(3-0.1*(v-vr)))
    def beta_func_m(v): return 4 * np.exp((vr-v)/18)
    def beta_func_n(v): return 0.125*np.exp((vr-v)/80)

    h[0] = alpha_func_h(vr) / (alpha_func_h(vr) + beta_func_h(vr))
    m[0] = alpha_func_m(vr) / (alpha_func_m(vr) + beta_func_m(vr))
    n[0] = alpha_func_n(vr) / (alpha_func_n(vr) + beta_func_n(vr))

    for i in range(1, t.shape[0]):

        I_na = g_na * h[i-1] * m[i-1]**3 * (v[i-1] - E_na)
        I_k = g_k * n[i-1]**4 * (v[i-1] - E_k)
        I_leak = g_leak * (v[i-1] - E_leak)

        # It looked like in the lecture code the C went missing.
        dvdt = I[i-1] - (I_na + I_k + I_leak) + I_inj[i-1]

        dhdt = alpha_func_h(v[i-1]) * (1 - h[i-1]) - \
            beta_func_h(v[i-1]) * h[i-1]
        dmdt = alpha_func_m(v[i-1]) * (1 - m[i-1]) - \
            beta_func_m(v[i-1]) * m[i-1]
        dndt = alpha_func_n(v[i-1]) * (1 - n[i-1]) - \
            beta_func_n(v[i-1]) * n[i-1]

        # delta t
        dt = t[i] - t[i-1]

        # Euler's update
        v[i] = v[i-1] + dvdt * dt
        h[i] = h[i-1] + dhdt * dt
        m[i] = m[i-1] + dmdt * dt
        n[i] = n[i-1] + dndt * dt
    return (t, v, I_inj)


# %% Question 1: Why do we care about Tau?
tau_values = [0.01, 0.05, 0.075, 1]
fig, ax = plt.subplots(len(tau_values), squeeze=True)

for index, value in enumerate(tau_values):
    t, v, _ = hh_model(value, 15)
    ax[index].plot(t, v)
    ax[index].set_title(f"Tau = {value}")
    ax[index].set_ylabel('v')
    ax[index].set_xlabel('t')
plt.show()

# As Tau increases, the liklihood of weird things happening due to high rate of change
# also increases. So as the rate of change increases, tau must decrease (unless you like errors)

# %% Question 2: What happens when we apply a DC Current?


def dc_inj(I): return lambda t: np.full(
    t.shape, I)  # Did someone say currying?


I_values = [0, 2]

fig, ax = plt.subplots(1, squeeze=True)

for index, value in enumerate(I_values):
    t, v, _ = hh_model(0.01, 15, inj=dc_inj(value))
    ax.plot(t, v, label=f"DC Current = {value}µA/cm^2")
ax.set_ylabel('v')
ax.set_xlabel('t')
ax.legend()
plt.show()

# It didn't really make sense to super impose the conductances, since they're constant.

# %% Question 2 It didn't really make sense to super impose the conductances,
#    since they're constant. I superimposed the currents instead.

# TODO: Refactor so I can superimpose the currents.

# %% Question 3: Inject 4µA/cm^2 during the middle third of the simulation.


def dc_middle_third(t):
    return np.concatenate((np.zeros(t.shape[0] // 3),
                           np.full(t.shape[0] // 3, 4.),
                           np.zeros(t.shape[0] // 3),
                           np.zeros(t.shape[0] % 3)))  # Gotta get 9999 to 10000!


fig, ax = plt.subplots(1, squeeze=True)

t, v, inj = hh_model(0.01, 100, inj=dc_middle_third)
ax.plot(t, v, label=f"Membrane Potential")
ax.set_ylabel('v')
ax.set_xlabel('t')

ax2 = ax.twinx()
ax2.plot(t, inj, color='red')

ax.legend()
plt.show()

# Hey cool, it looks like it makes our neuron spike more often!

# %% Question 4: Show that injecting a DC Current is like changing E Leak

fig, ax = plt.subplots(1, squeeze=True)

E_leak_values = [10.6, 21.2]

for index, value in enumerate(E_leak_values):
    t, v, _ = hh_model(0.01, 15, E_leak=value)
    ax.plot(t, v, label=f"E_leak = {value} mV")
ax.set_ylabel('v')
ax.set_xlabel('t')
ax.legend()
plt.show()

# Looks familiar, doesn't it?

# %%
