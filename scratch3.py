import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def update_vis(o, i):

    vis_amp = 2000
    vis_onset = 300
    vis_offset = 900

    if i >= vis_onset and i < vis_offset:
        o[i + 1] = vis_amp
    else:
        o[i + 1] = 0


def update_pf(o, i):

    pf_amp = 2000
    pf_decay = 0.99
    pf_onset = 300
    pf_offset = 400

    if i >= pf_onset and i < pf_offset:
        o_pf[i + 1] = pf_amp

    elif i >= pf_offset:
        o_pf[i + 1] = pf_decay * o_pf[i]


def update_tan(v, u, o, I_v, I_u, i):

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

    v[0] = vr

    v[i + 1] = v[i] + tau * (k * (v[i] - vr) * (v[i] - vt) - u[i] + I_v[i]) / C
    u[i + 1] = u[i] + tau * a * (b * (v[i] - vr) - u[i] + I_u[i])
    o[i + 1] = o[i] + alpha * (np.heaviside(v[i] - vt, vt) - o[i])

    if v[i + 1] >= vpeak:
        v[i] = vpeak
        v[i + 1] = c
        u[i + 1] = u[i + 1] + d


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

    v[i + 1] = v[i] + tau * (k * (v[i] - vr) * (v[i] - vt) - u[i] + I[i]) / C
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


n_simulations = 1
n_trials = 300
n_steps = 1200

# neurotransmitter release
alpha = 0.025

# vis
o_vis = np.zeros(n_steps)

# pf
o_pf = np.zeros(n_steps)

# msn d1
v_d1 = np.zeros((n_steps, n_trials))
u_d1 = np.zeros((n_steps, n_trials))
o_d1 = np.zeros((n_steps, n_trials))
d1_base = 100.0

# msn d2
v_d2 = np.zeros((n_steps, n_trials))
u_d2 = np.zeros((n_steps, n_trials))
o_d2 = np.zeros((n_steps, n_trials))
d2_base = 100.0

# # TAN
# v_tan = np.zeros((n_steps, n_trials))
# u_tan = np.zeros((n_steps, n_trials))
# o_tan = np.zeros((n_steps, n_trials))
# tan_base = 950.0

# globus pallidus internal
v_gpi = np.zeros((n_steps, n_trials))
o_gpi = np.zeros((n_steps, n_trials))
gpi_base = 300

# globus pallidus external
v_gpe = np.zeros((n_steps, n_trials))
o_gpe = np.zeros((n_steps, n_trials))
gpe_base = 72.5

# thalamus
v_th = np.zeros((n_steps, n_trials))
o_th = np.zeros((n_steps, n_trials))
th_base = 300

# motor
v_m1 = np.zeros((n_steps, n_trials))
o_m1 = np.zeros((n_steps, n_trials))
m1_base = 0.0

# w_pf_tan = 0.2 * np.ones((n_trials, n_simulations))
w_vis_d1 = 0.2 * np.ones((n_trials, n_simulations))
w_vis_d2 = 0.2 * np.ones((n_trials, n_simulations))
# w_tan_msn = 6000
w_d1_gpi = 2000.0
w_gpi_th = 2000.0
w_th_m1 = 2000.0
w_d2_gpe = 2000.0
w_gpe_gpi = 1000.0

resp_thresh = 0.55
resp = np.zeros((n_trials, n_simulations))
r = np.zeros((n_trials, n_simulations))
pr = np.zeros((n_trials, n_simulations))
delta = np.zeros((n_trials, n_simulations))
pr_alpha = 0.1

w_ltp_d1 = 1e-10
w_ltd_d1 = 5e-12
w_ltp_d2 = 1e-11
w_ltd_d2 = 1e-11
# w_ltp_tan = 3e-3
# w_ltd_tan = 5e-3

for k in range(n_simulations):
    for j in range(n_trials - 1):
        print(k, j)
        for i in range(n_steps - 1):

            # update_pf(o_pf, i)

            update_vis(o_vis, i)

            # update_tan(v_tan[:, j], u_tan[:, j], o_tan[:, j],
            #            tan_base + w_pf_tan[j, k] * o_pf, w_pf_tan[j, k] * o_pf,
            #            i)

            update_msn(v_d1[:, j], u_d1[:, j], o_d1[:, j],
                       w_vis_d1[j, k] * o_vis + d1_base, i)

            update_msn(v_d2[:, j], u_d2[:, j], o_d2[:, j],
                       w_vis_d2[j, k] * o_vis + d2_base, i)

            update_qif(v_gpe[:, j], o_gpe[:, j],
                       gpe_base - w_d2_gpe * o_d2[:, j], i)

            update_qif(
                v_gpi[:, j], o_gpi[:, j],
                gpi_base - w_d1_gpi * o_d1[:, j] - w_gpe_gpi * o_gpe[:, j], i)

            update_qif(v_th[:, j], o_th[:, j],
                       th_base - w_gpi_th * o_gpi[:, j], i)
            update_qif(
                v_m1[:, j], o_m1[:, j], m1_base + w_th_m1 * o_th[:, j] +
                np.random.normal(10, 200, n_steps), i)

            if o_m1[i, j] > resp_thresh:
                resp[j, k] = 1

        if resp[j, k] == 1:
            if j > 100 and j < 200:
                r[j, k] = 0
            else:
                r[j, k] = 1

        delta[j, k] = r[j, k] - pr[j, k]
        pr[j + 1, k] = pr[j, k] + pr_alpha * delta[j, k]

        sum_vis = o_vis.sum()
        sum_d1 = o_d1.sum()
        sum_d2 = o_d1.sum()
        # sum_pf = o_pf.sum()
        # sum_tan = o_tan[o_pf > 0].sum()

        if delta[j, k] >= 0:
            w_vis_d1[j + 1, k] = w_vis_d1[
                j, k] + w_ltp_d1 * sum_vis * sum_d1 * delta[j, k]
            w_vis_d2[j + 1, k] = w_vis_d2[
                j, k] - w_ltp_d2 * sum_vis * sum_d2 * delta[j, k]
            # w_pf_tan[j + 1, k] = w_pf_tan[
            #     j, k] + w_ltp_tan * sum_pf * sum_tan * delta[j, k]

        elif delta[j, k] < 0:
            w_vis_d1[j + 1, k] = w_vis_d1[
                j, k] + w_ltd_d1 * sum_vis * sum_d1 * delta[j, k]
            w_vis_d2[j + 1, k] = w_vis_d2[
                j, k] - w_ltd_d2 * sum_vis * sum_d2 * delta[j, k]
            # w_pf_tan[j + 1, k] = w_pf_tan[
            #     j, k] + w_ltd_tan * sum_pf * sum_tan * delta[j, k]

        w_vis_d1 = np.clip(w_vis_d1, 0, 1)
        w_vis_d2 = np.clip(w_vis_d2, 0, 1)
        # w_pf_tan = np.clip(w_pf_tan, 0, 1)

trial = 290
fig, ax = plt.subplots(nrows=6, ncols=2)
ax[0, 0].plot(np.arange(0, n_steps, 1), o_vis)
# ax[0, 0].plot(np.arange(0, n_steps, 1), o_pf)
ax[0, 0].set_ylabel('Input')

ax[0, 1].plot(np.arange(0, n_steps, 1), o_vis)
# ax[0, 1].plot(np.arange(0, n_steps, 1), o_pf)
ax[0, 1].set_ylabel('Input')

ax[1, 0].plot(np.arange(0, n_steps, 1), v_d1[:, trial], 'C0')
ax[1, 0].set_ylabel('Membrane V')
ax1 = ax[1, 0].twinx()
ax1.plot(np.arange(0, n_steps, 1), o_d1[:, trial], 'C1')
ax1.set_ylabel('Output')
ax1.yaxis.label.set_color('C1')

ax[1, 1].plot(np.arange(0, n_steps, 1), v_d2[:, trial], 'C0')
ax[1, 1].set_ylabel('Membrane V')
ax1 = ax[1, 1].twinx()
ax1.plot(np.arange(0, n_steps, 1), o_d2[:, trial], 'C1')
ax1.set_ylabel('Output')
ax1.yaxis.label.set_color('C1')

ax[2, 0].plot(np.arange(0, n_steps, 1), v_gpi[:, trial], 'C0')
ax[2, 0].set_ylabel('Membrane V')
ax1 = ax[2, 0].twinx()
ax1.plot(np.arange(0, n_steps, 1), o_gpi[:, trial], 'C1')
ax1.set_ylabel('Output')
ax1.yaxis.label.set_color('C1')

ax[2, 1].plot(np.arange(0, n_steps, 1), v_gpe[:, trial], 'C0')
ax[2, 1].set_ylabel('Membrane V')
ax1 = ax[2, 1].twinx()
ax1.plot(np.arange(0, n_steps, 1), o_gpe[:, trial], 'C1')
ax1.set_ylabel('Output')
ax1.yaxis.label.set_color('C1')

ax[3, 0].plot(np.arange(0, n_steps, 1), v_th[:, trial], 'C0')
ax[3, 0].set_ylabel('Membrane V')
ax1 = ax[3, 0].twinx()
ax1.plot(np.arange(0, n_steps, 1), o_th[:, trial], 'C1')
ax1.set_ylabel('Output')
ax1.yaxis.label.set_color('C1')

ax[4, 0].plot(np.arange(0, n_steps, 1), v_m1[:, trial], 'C0')
ax[4, 0].set_ylabel('Membrane V')
ax1 = ax[4, 0].twinx()
ax1.plot(np.arange(0, n_steps, 1), o_m1[:, trial], 'C1')
ax1.set_ylabel('Output')
ax1.yaxis.label.set_color('C1')

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(nrows=3, ncols=1)
ax[0].plot(np.arange(0, n_trials), resp.mean(axis=1))
ax[0].set_ylabel('Response')

ax[1].plot(np.arange(0, n_trials), r.mean(axis=1))
ax[1].plot(np.arange(0, n_trials), pr.mean(axis=1))
ax[1].plot(np.arange(0, n_trials), delta.mean(axis=1))
ax[1].legend(['r', 'pr', 'delta'])

ax[2].plot(np.arange(0, n_trials), w_vis_d1.mean(axis=1))
ax[2].set_ylabel('w_vis_d1')
ax1 = ax[2].twinx()
ax1.plot(np.arange(0, n_trials), w_vis_d2.mean(axis=1), 'C1')
ax1.set_ylabel('w_vis_d2')

plt.tight_layout()
plt.show()
