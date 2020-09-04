import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# add trials, and motor responses via motor noise
# The direct pathway through the basal ganglia
def update_msn(v, u, o, I, i):
    tau = 1
    C = 50
    vr = -80
    vt = -25
    k = 1
    a = 0.01
    b = -20
    c = -55
    d = 150
    vpeak = 40

    v[0] = vr

    v[i + 1] = v[i] + tau * (k * (v[i] - vr) *
                             (v[i] - vt) - u[i] + I[i]) / C
    u[i + 1] = u[i] + tau * a * (b * (v[i] - vr) - u[i])
    o[i + 1] = o[i] + alpha * (np.heaviside(v[i] - vt, vt) - o[i])

    if v[i + 1] >= vpeak:
        v[i] = vpeak
        v[i + 1] = c
        u[i + 1] = u[i + 1] + d


def update_qif(v, o, I, i):
    tau = 1
    C = 25
    vr = -60
    vt = -40
    k = 0.7
    c = -50
    vpeak = 35

    v[0] = vr

    v[i + 1] = v[i] + tau * (k * (v[i] - vr) * (v[i] - vt) + I[i]) / C
    o[i + 1] = o[i] + alpha * (np.heaviside(v[i] - vt, vt) - o[i])
    if v[i + 1] >= vpeak:
        v[i] = vpeak
        v[i + 1] = c


n_simulations = 100
n_trials = 300
n_steps = 900

input_amp = 2000
E = np.concatenate((np.zeros(n_steps // 3), input_amp * np.ones(n_steps // 3),
                    np.zeros(n_steps // 3)))

# neurotransmitter release
alpha = 0.01
beta = 100.0

# msn
v_d1 = np.zeros((n_steps, n_trials))
u_d1 = np.zeros((n_steps, n_trials))
o_d1 = np.zeros((n_steps, n_trials))
d1_base = 0.0

# globus pallidus
v_gp = np.zeros((n_steps, n_trials))
o_gp = np.zeros((n_steps, n_trials))
gp_base = 300

# thalamus
v_th = np.zeros((n_steps, n_trials))
o_th = np.zeros((n_steps, n_trials))
th_base = 300

# motor
v_m1 = np.zeros((n_steps, n_trials))
o_m1 = np.zeros((n_steps, n_trials))
m1_base = 0.0

w_vis_d1 = 0.25 * np.ones((n_trials, n_simulations))
w_d1_gp = 2000.0
w_gp_th = 2000.0
w_th_m1 = 2000.0

resp_thresh = 0.385
resp = np.zeros((n_trials, n_simulations))
r = np.zeros((n_trials, n_simulations))
pr = np.zeros((n_trials, n_simulations))
delta = np.zeros((n_trials, n_simulations))
pr_alpha = 0.05

w_ltp = 0.2
w_ltd = 0.2

for k in range(n_simulations):
    print(k)
    for j in range(n_trials - 1):
        for i in range(n_steps - 1):
            update_msn(v_d1[:, j], u_d1[:, j], o_d1[:, j], w_vis_d1[j, k] * E, i)
            update_qif(v_gp[:, j], o_gp[:, j], gp_base - w_d1_gp * o_d1[:, j], i)
            update_qif(v_th[:, j], o_th[:, j], th_base - w_gp_th * o_gp[:, j], i)
            update_qif(
                v_m1[:, j], o_m1[:, j],
                m1_base + w_th_m1 * o_th[:, j] + np.random.normal(0, 100, n_steps), i)
            if o_m1[i, j] > resp_thresh:
                resp[j, k] = 1

        if resp[j, k] == 1:
            if j > 100 and j < 200:
                r[j, k] = 0
            else:
                r[j, k] = 1

            delta[j, k] = r[j, k] - pr[j, k]
            pr[j + 1, k] = pr[j, k] + pr_alpha * delta[j, k]

        if delta[j, k] > 0:
            w_vis_d1[j + 1, k] = w_vis_d1[j, k] + w_ltp * delta[j, k]
        elif delta[j, k] < 0:
            w_vis_d1[j + 1, k] = w_vis_d1[j, k] + w_ltd * delta[j, k]

fig, ax = plt.subplots(nrows=3, ncols=1)
ax[0].plot(np.arange(0, n_trials), resp.mean(axis=1))
ax[0].set_ylabel('Response')

ax[1].plot(np.arange(0, n_trials), r.mean(axis=1))
ax[1].plot(np.arange(0, n_trials), pr.mean(axis=1))
ax[1].plot(np.arange(0, n_trials), delta.mean(axis=1))
ax[1].set_ylabel('r, pr, delta')

ax[2].plot(np.arange(0, n_trials), w_vis_d1.mean(axis=1))
ax[2].set_ylabel('w_vis_d1')

plt.tight_layout()
plt.show()
