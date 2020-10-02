import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def update_vis(o, x, y, i):

    vis_amp = 2000
    vis_onset = 300
    vis_width = 25

    if i >= vis_onset:
        o[i + 1] = vis_amp

        vis_dist_x = 0.0
        vis_dist_y = 0.0
        for ii in range(0, dim):
            for jj in range(0, dim):
                vis_dist_x = x - ii
                vis_dist_y = y - jj

                o[i + 1, jj + ii * dim] = vis_amp * np.exp(
                    -(vis_dist_x**2 + vis_dist_y**2) / vis_width)

    else:
        o[i + 1, :] = 0


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

    v[i + 1] = v[i] + tau * (k * (v[i] - vr) * (v[i] - vt) - u[i] + I_v) / C
    u[i + 1] = u[i] + tau * a * (b * (v[i] - vr) - u[i] + I_u)
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

    v[i + 1] = v[i] + tau * (k * (v[i] - vr) * (v[i] - vt) - u[i] + I) / C
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

    v[i + 1] = v[i] + tau * (k * (v[i] - vr) * (v[i] - vt) + I) / C
    o[i + 1] = o[i] + alpha * (np.heaviside(v[i] - vt, vt) - o[i])
    if v[i + 1] >= vpeak:
        v[i] = vpeak
        v[i + 1] = c


def gen_stim(mean, cov, cat_lab, num_stim):

    x = []
    y = []
    cat = []

    for i in range(len(mean)):
        m = mean[i]
        c = cov[i]

        xy = np.random.multivariate_normal(m, c, num_stim[i])
        lab = [cat_lab[i]] * num_stim[i]

        x = np.append(x, xy[:, 0])
        y = np.append(y, xy[:, 1])
        cat = np.append(cat, lab)

    return {'cat': cat, 'x': x, 'y': y}


# Define Crossley et al. (2013) categories
mean_x = [72, 100, 100, 128]
mean_y = [100, 128, 72, 100]
cov = [[100, 0], [0, 100]]

mean_x = [x / 2.0 for x in mean_x]
mean_y = [x / 2.0 for x in mean_y]
cov = [[x[0] / 2.0, x[1] / 2.0] for x in cov]

n = 5
mean = [(mean_x[i], mean_y[i]) for i in range(len(mean_x))]

stimuli_A = gen_stim(mean, [cov, cov, cov, cov], [1, 2, 3, 4], [n, n, n, n])
stimuli_B = gen_stim(mean, [cov, cov, cov, cov], [2, 3, 4, 1], [n, n, n, n])
stimuli_A2 = gen_stim(mean, [cov, cov, cov, cov], [1, 2, 3, 4], [n, n, n, n])

stimuli = {
    'cat': np.append(stimuli_A['cat'], [stimuli_B['cat'], stimuli_A2['cat']]),
    'x': np.append(stimuli_A['x'], [stimuli_B['x'], stimuli_A2['x']]),
    'y': np.append(stimuli_A['y'], [stimuli_B['y'], stimuli_A2['y']])
}

# plot stimuli
x = stimuli_A['x']
y = stimuli_A['y']
cat = stimuli_A['cat']

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(x[cat == 1], y[cat == 1], '.r')
ax.plot(x[cat == 2], y[cat == 2], '.b')
ax.plot(x[cat == 3], y[cat == 3], '.g')
ax.plot(x[cat == 4], y[cat == 4], '.k')
ax.set_xlim([0, 100])
ax.set_ylim([0, 100])
plt.show()

n_simulations = 1
n_trials = n * 4
n_steps = 1200

dim = 100

# neurotransmitter release
alpha = 0.025

# vis
o_vis = np.zeros((n_steps, dim**2))

# pf
o_pf = np.zeros(n_steps)

# TAN
v_tan = np.zeros((n_steps, n_trials))
u_tan = np.zeros((n_steps, n_trials))
o_tan = np.zeros((n_steps, n_trials))
tan_base = 950.0

# msn
v_d1 = np.zeros((n_steps, n_trials, 4))
u_d1 = np.zeros((n_steps, n_trials, 4))
o_d1 = np.zeros((n_steps, n_trials, 4))
d1_base = 100.0

# globus pallidus
v_gp = np.zeros((n_steps, n_trials, 4))
o_gp = np.zeros((n_steps, n_trials, 4))
gp_base = 300

# thalamus
v_th = np.zeros((n_steps, n_trials, 4))
o_th = np.zeros((n_steps, n_trials, 4))
th_base = 300

# motor
v_m1 = np.zeros((n_steps, n_trials, 4))
o_m1 = np.zeros((n_steps, n_trials, 4))
m1_base = 0.0

w_pf_tan = 0.2 * np.ones((n_trials, n_simulations))
w_vis_d1 = 0.2 * np.ones((n_trials, n_simulations, dim**2, 4))
w_tan_msn = 10000
w_d1_gp = 2000.0
w_gp_th = 2000.0
w_th_m1 = 2000.0

# responses
resp_thresh = 0.475
resp = -1 * np.ones((n_trials, n_simulations))

# rewards et al
r = np.zeros((n_trials, n_simulations))
pr = np.zeros((n_trials, n_simulations))
delta = np.zeros((n_trials, n_simulations))
pr_alpha = 0.01

# learning weights
w_ltp_d1 = 1e-8
w_ltd_d1 = 1e-11
w_ltp_tan = 3e-6
w_ltd_tan = 5e-6

for k in range(n_simulations):
    for j in range(n_trials - 1):
        print(k, j)

        cat = stimuli['cat'][j]
        x = stimuli['x'][j]
        y = stimuli['y'][j]

        for i in range(n_steps - 1):

            # TODO: inefficient for pf and vis -- change later
            update_pf(o_pf, i)

            update_vis(o_vis, x, y, i)

            update_tan(v_tan[:, j], u_tan[:, j], o_tan[:, j],
                       tan_base + w_pf_tan[j, k] * o_pf[i], w_pf_tan[j, k] * o_pf[i],
                       i)

            # Compute unit activation via dot product
            act_A = np.inner(o_vis[i, :], w_vis_d1[j, k, :, 0])
            act_B = np.inner(o_vis[i, :], w_vis_d1[j, k, :, 1])
            act_C = np.inner(o_vis[i, :], w_vis_d1[j, k, :, 2])
            act_D = np.inner(o_vis[i, :], w_vis_d1[j, k, :, 3])

            act_A += -w_tan_msn * o_tan[i, j] + d1_base
            act_B += -w_tan_msn * o_tan[i, j] + d1_base
            act_C += -w_tan_msn * o_tan[i, j] + d1_base
            act_D += -w_tan_msn * o_tan[i, j] + d1_base

            input_msn = np.array([act_A, act_B, act_C, act_D])

            for ii in range(4):

                lateral_inhibition_ind = np.array([0, 1, 2, 3])
                lateral_inhibition_ind = lateral_inhibition_ind[
                    lateral_inhibition_ind != ii]
                lateral_inhibition = np.sum(o_d1[i, j, lateral_inhibition_ind])

                update_msn(v_d1[:, j, ii], u_d1[:, j, ii], o_d1[:, j, ii],
                           input_msn[ii] - lateral_inhibition, i)

                update_qif(v_gp[:, j, ii], o_gp[:, j, ii],
                           gp_base - w_d1_gp * o_d1[i, j, ii], i)

                update_qif(v_th[:, j, ii], o_th[:, j, ii],
                           th_base - w_gp_th * o_gp[i, j, ii], i)
                update_qif(
                    v_m1[:, j, ii], o_m1[:, j, ii],
                    m1_base + w_th_m1 * o_th[i, j, ii] +
                    np.random.normal(10, 100), i)

                if o_m1[i, j, ii] > resp_thresh:
                    resp[j, k] = ii

        # If no response by end of trial, force a response from the max at end
        if resp[j, k] == -1:
            act_array = o_m1[-1, j, :]
            act_sort_ind = np.argsort(act_array, 0)
            resp[j, k] = act_sort_ind[-1] + 1

        if resp[j, k] == cat:
            r[j, k] = 1
        else:
            r[j, k] = -1

        delta[j, k] = r[j, k] - pr[j, k]
        pr[j + 1, k] = pr[j, k] + pr_alpha * delta[j, k]

        # compute pre and post synaptic activities
        sum_vis = o_vis.sum()
        sum_d1 = np.array([
            o_d1[:, j, 0].sum(), o_d1[:, j, 1].sum(), o_d1[:, j, 2].sum(),
            o_d1[:, j, 3].sum()
        ])

        sum_pf = o_pf.sum()
        sum_tan = o_tan[o_pf > 0].sum()

        # update synaptic weights
        if delta[j, k] >= 0:
            w_pf_tan[j + 1, k] = w_pf_tan[
                j, k] + w_ltp_tan * sum_pf * sum_tan * delta[j, k]

            for ii in range(dim**2):
                w_vis_d1[j + 1, k, ii] = w_vis_d1[
                    j, k, ii] + w_ltp_d1 * sum_vis * sum_d1 * delta[j, k]

        elif delta[j, k] < 0:
            w_pf_tan[j + 1, k] = w_pf_tan[
                j, k] + w_ltd_tan * sum_pf * sum_tan * delta[j, k]

            for ii in range(dim**2):
                w_vis_d1[j + 1, k, ii] = w_vis_d1[
                    j, k, ii] + w_ltd_d1 * sum_vis * sum_d1 * delta[j, k]

        w_pf_tan = np.clip(w_pf_tan, 0, 1)
        w_vis_d1 = np.clip(w_vis_d1, 0, 1)
