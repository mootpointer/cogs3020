import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

#################
# parameters
#################

tau = 1
spike_decay = 0.005

n_simulations = 1
n_trials = 4  # must be even number
n_steps = 10000
n_channels = 2

# vis parameters
vis_dim = 10
vis_amp = 2500
vis_onset = 2000
vis_offset = 6000
vis_width = 0.25

# pf parameters
pf_amp = 2000
pf_decay = 1 - 2e-3
pf_onset = 2000
pf_offset = 2050

# connection weights
w_lat = 1e5
w_vis_msn_d1_init = 0.1
w_pf_tan = 0.1
w_tan_msn_d1 = 2000
w_msn_d1_gpi = 3e4
w_gpi_th = 500
w_th_pm = 700
w_da_tan = 100
w_tan_da = 400
w_delta_da = 500

# learning rates
w_ltp_msn_d1 = 1e-8
w_ltd_msn_d1 = 1e-8
w_ltp_tan = 1e-1
w_ltd_tan = 1e-12 * 0

# responses
resp_thresh = 1
resp = -1
response_deadline = 6000

# rewards et al
r = 0
pr = 0
delta = 0
pr_alpha = 0.01

#################
# allocate arrays
#################
resp_rec = np.zeros((n_simulations, n_trials))
r_rec = np.zeros((n_simulations, n_trials))
delta_rec = np.zeros((n_simulations, n_trials))
pr_rec = np.zeros((n_simulations, n_trials))

w_vis_msn_d1 = np.random.normal(w_vis_msn_d1_init, 0.01,
                                (vis_dim**2, n_channels))
o_vis = np.zeros((n_steps, vis_dim**2))

o_pf_v = np.zeros(n_steps)
o_pf_u = np.zeros(n_steps)

v_tan = np.zeros((n_steps, 1))
u_tan = np.zeros((n_steps, 1))
o_tan = np.zeros((n_steps, 1))
base_tan = 750.0

v_msn_d1 = np.zeros((n_steps, n_channels))
u_msn_d1 = np.zeros((n_steps, n_channels))
o_msn_d1 = np.zeros((n_steps, n_channels))
base_msn_d1 = 250

v_gpi = np.zeros((n_steps, n_channels))
o_gpi = np.zeros((n_steps, n_channels))
base_gpi = 300

v_th = np.zeros((n_steps, n_channels))
o_th = np.zeros((n_steps, n_channels))
base_th = 200

v_pm = np.zeros((n_steps, n_channels))
o_pm = np.zeros((n_steps, n_channels))
base_pm = 10

v_da = np.zeros((n_steps, 1))
o_da = np.zeros((n_steps, 1))
base_da = 150


#################
# functions
#################
def plot_learning():

    fig, ax = plt.subplots(2, 1, squeeze=True)
    ax[0].plot(resp_rec.mean(axis=0))
    ax[1].plot(r_rec.mean(axis=0))
    ax[1].plot(delta_rec.mean(axis=0))
    ax[1].plot(pr_rec.mean(axis=0))
    plt.show()


def plot_network():

    av = 0.1
    ao = 1
    fig, ax = plt.subplots(7, 2, squeeze=False, figsize=(10, 6))
    ax[0, 0].plot(o_pf_u, alpha=ao)
    ax[1, 0].plot(v_da, alpha=av)
    ax[1, 0].twinx().plot(o_da, alpha=ao)
    ax[2, 0].plot(v_tan, alpha=av)
    ax[2, 0].twinx().plot(o_tan, alpha=ao)
    ax[3, 0].plot(v_msn_d1, alpha=av)
    ax[3, 0].twinx().plot(o_msn_d1, alpha=ao)
    ax[4, 0].plot(v_gpi, alpha=av)
    ax[4, 0].twinx().plot(o_gpi, alpha=ao)
    ax[5, 0].plot(v_th, alpha=av)
    ax[5, 0].twinx().plot(o_th, alpha=ao)
    ax[6, 0].plot(v_pm, alpha=av)
    ax[6, 0].twinx().plot(o_pm, alpha=ao)

    ax[0, 1].imshow(np.reshape(vis_act, (vis_dim, vis_dim)))
    for i in range(n_channels):
        ax[i + 1, 1].imshow(np.reshape(w_vis_msn_d1[:, i], (vis_dim, vis_dim)),
                            vmin=0,
                            vmax=1)

    plt.tight_layout()
    plt.show()


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


def gen_cat_2():

    mean_x = [3, 7]
    mean_y = [7, 3]
    v = 0.01
    cv = 0.1
    cov = [[v, cv], [cv, v]]

    n = n_trials // 2
    mean = [(mean_x[i], mean_y[i]) for i in range(len(mean_x))]
    cov = [cov, cov]
    nn = [n, n]
    cat = [1, 2]
    stimuli = gen_stim(mean, cov, cat, nn)

    # cat = [1, 2]
    # stimuli_A = gen_stim(mean, cov, cat, nn)
    # cat = [2, 1]
    # stimuli_B = gen_stim(mean, cov, cat, nn)
    # cat = [1, 2]
    # stimuli_A2 = gen_stim(mean, cov, cat, nn)
    # stimuli = {
    #     'cat': np.append(stimuli_A['cat'],
    #                      [stimuli_B['cat'], stimuli_A2['cat']]),
    #     'x': np.append(stimuli_A['x'], [stimuli_B['x'], stimuli_A2['x']]),
    #     'y': np.append(stimuli_A['y'], [stimuli_B['y'], stimuli_A2['y']])
    # }

    stimuli = pd.DataFrame(stimuli)
    stimuli = stimuli.sample(frac=1).reset_index(drop=True)

    # plot stimuli
    x = stimuli['x']
    y = stimuli['y']
    cat = stimuli['cat']

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.plot(x[cat == 1], y[cat == 1], '.r')
    # ax.plot(x[cat == 2], y[cat == 2], '.b')
    # ax.plot(x[cat == 3], y[cat == 3], '.g')
    # ax.plot(x[cat == 4], y[cat == 4], '.k')
    # # ax.set_xlim([0, 10])
    # # ax.set_ylim([0, 10])
    # plt.show()

    return stimuli


def gen_cat_4():

    # Define Crossley et al. (2013) categories
    mean_x = [7.2, 10.0, 10.0, 12.8]
    mean_y = [10.0, 12.8, 7.2, 10.0]
    cov = [[0.2, 0], [0, 0.2]]

    mean_x = [x / 2.0 for x in mean_x]
    mean_y = [x / 2.0 for x in mean_y]
    cov = [[x[0] / 2.0, x[1] / 2.0] for x in cov]

    n = n_trials // 4
    mean = [(mean_x[i], mean_y[i]) for i in range(len(mean_x))]

    stimuli_A = gen_stim(mean, [cov, cov, cov, cov], [1, 2, 3, 4],
                         [n, n, n, n])
    stimuli_B = gen_stim(mean, [cov, cov, cov, cov], [2, 3, 4, 1],
                         [n, n, n, n])
    stimuli_A2 = gen_stim(mean, [cov, cov, cov, cov], [1, 2, 3, 4],
                          [n, n, n, n])

    stimuli = {
        'cat': np.append(stimuli_A['cat'],
                         [stimuli_B['cat'], stimuli_A2['cat']]),
        'x': np.append(stimuli_A['x'], [stimuli_B['x'], stimuli_A2['x']]),
        'y': np.append(stimuli_A['y'], [stimuli_B['y'], stimuli_A2['y']])
    }

    stimuli = pd.DataFrame(stimuli)
    stimuli = stimuli.sample(frac=1).reset_index(drop=True)

    # plot stimuli
    x = stimuli_A['x']
    y = stimuli_A['y']
    cat = stimuli_A['cat']

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.plot(x[cat == 1], y[cat == 1], '.r')
    # ax.plot(x[cat == 2], y[cat == 2], '.b')
    # ax.plot(x[cat == 3], y[cat == 3], '.g')
    # ax.plot(x[cat == 4], y[cat == 4], '.k')
    # # ax.set_xlim([0, 10])
    # # ax.set_ylim([0, 10])
    # plt.show()

    return stimuli


def update_vis(x, y):
    global vis_act

    o = o_vis

    xx, yy = np.mgrid[0:vis_dim:1, 0:vis_dim:1]
    pos = np.dstack((xx, yy))
    rf = multivariate_normal([x, y], [[vis_width, 0], [0, vis_width]])
    vis_act = rf.pdf(pos)
    o_vis[vis_onset:vis_offset, :] = vis_act.flatten() * vis_amp


def update_pf():

    o_pf_v[pf_onset:pf_offset] = pf_amp
    o_pf_u[pf_onset:pf_offset] = pf_amp
    for i in range(pf_offset, n_steps):
        o_pf_u[i] = pf_decay * o_pf_u[i - 1]

    pf_onset_2 = 6000
    pf_offset_2 = 6050
    o_pf_v[pf_onset_2:pf_offset_2] = pf_amp
    o_pf_u[pf_onset_2:pf_offset_2] = pf_amp
    for i in range(pf_offset_2, n_steps):
        o_pf_u[i] = pf_decay * o_pf_u[i - 1]


def update_tan(i):

    v = v_tan
    u = u_tan
    o = o_tan
    base = base_tan

    I_v = base_tan + w_pf_tan * o_pf_v[i]
    I_u = w_pf_tan * o_pf_u[i] + w_da_tan * o_da[i, 0]

    C = 100.0
    vr = -75.0
    vt = -45.0
    k = 1.2
    a = 0.01
    b = 5.0
    c = -56.0
    d = 130.0
    vpeak = 60.0
    v[0, :] = vr

    v[i + 1, :] = v[i, :] + tau * (k * (v[i, :] - vr) *
                                   (v[i, :] - vt) - u[i, :] + I_v) / C

    u[i + 1, :] = u[i, :] + tau * a * (b * (v[i, :] - vr) - u[i, :] + I_u)

    o[i + 1, :] = o[i, :] + spike_decay * (np.heaviside(v[i, :] - vt, vt) -
                                           o[i, :])

    for ii in range(n_channels):
        if v[i + 1] >= vpeak:
            v[i] = vpeak
            v[i + 1] = c
            u[i + 1] = u[i + 1] + d


def update_msn_d1(i):

    v = v_msn_d1
    u = u_msn_d1
    o = o_msn_d1
    base = base_msn_d1

    # compute inputs
    lat = np.outer(o[i, :], o[i, :]) * ~np.eye(n_channels, dtype=np.bool)
    lat = lat.sum(axis=0, keepdims=True) * w_lat
    oo = o[i, :]
    lat[0, oo > 0] = lat[0, oo > 0] / oo[oo > 0]

    # TODO: no need to compute inner every step... only per trial
    vis = np.inner(o_vis[i, :], w_vis_msn_d1.T)
    vis = np.clip(vis - w_tan_msn_d1 * o_tan[i, :], 0, None)

    noise = np.random.normal(base, 10, n_channels)

    I = noise + vis - lat

    C = 50
    vr = -80
    vt = -25
    k = 1
    a = 0.01
    b = -20
    c = -55
    d = 150
    vpeak = 40
    v[0, :] = vr

    v[i + 1, :] = v[i, :] + tau * (k * (v[i, :] - vr) *
                                   (v[i, :] - vt) - u[i, :] + I) / C
    u[i + 1, :] = u[i, :] + tau * a * (b * (v[i, :] - vr) - u[i, :])
    o[i + 1, :] = o[i, :] + spike_decay * (np.heaviside(v[i, :] - vt, vt) -
                                           o[i, :])

    for ii in range(n_channels):
        if v[i + 1, ii] >= vpeak:
            v[i, ii] = vpeak
            v[i + 1, ii] = c
            u[i + 1, ii] = u[i + 1, ii] + d


def update_gpi(i):

    v = v_gpi
    o = o_gpi
    base = base_gpi
    I = base - w_msn_d1_gpi * o_msn_d1[i, :]
    update_qif(v, o, I, i)


def update_th(i):

    v = v_th
    o = o_th
    base = base_th
    I = base - w_gpi_th * o_gpi[i, :]
    update_qif(v, o, I, i)


def update_pm(i):

    v = v_pm
    o = o_pm
    base = base_pm
    I = w_th_pm * o_th[i, :] + np.random.normal(base, 100, n_channels)
    update_qif(v, o, I, i)


def update_da(i):

    v = v_da
    o = o_da
    base = base_da
    I = base - w_tan_da * o_tan[i, :] + w_delta_da * delta
    update_qif(v, o, I, i)


def update_qif(v, o, I, i):

    C = 25
    vr = -60
    vt = -40
    k = 0.7
    c = -50
    vpeak = 35
    v[0, :] = vr

    v[i + 1, :] = v[i, :] + tau * (k * (v[i, :] - vr) * (v[i, :] - vt) + I) / C
    o[i + 1, :] = o[i, :] + spike_decay * (np.heaviside(v[i, :] - vt, vt) -
                                           o[i, :])

    for ii in range(v.shape[1]):
        if v[i + 1, ii] >= vpeak:
            v[i, ii] = vpeak
            v[i + 1, ii] = c


def update_response(i):
    global resp

    if i > vis_onset:
        if np.any(o_pm[i, :] > resp_thresh):
            act_array = o_pm[i, :]
            act_sort_ind = np.argsort(act_array, 0)
            resp = act_sort_ind[-1] + 1
            o_vis[i:, :] *= 0

    # If no response by end of trial, force it
    if i == (response_deadline) and resp == -1:
        act_array = o_pm[n_steps - 2, :]
        act_sort_ind = np.argsort(act_array, 0)
        resp = act_sort_ind[-1] + 1
        # resp = np.random.choice([1, 2])


def update_reward():
    global r, delta, pr

    if resp != -1:
        r = 1 if resp == cat else -1
        delta = r - pr


def update_weights():
    global w_pf_tan, w_vis_msn_d1, pr

    # NOTE: Be careful about updating this every time step
    pr += pr_alpha * delta

    # compute pre and post synaptic activities
    sum_vis = o_vis.sum(0)
    sum_msn_d1 = o_msn_d1.sum(0)
    sum_pf = o_pf_v.sum()
    sum_tan = o_tan[pf_onset:pf_offset].sum()

    # update synaptic weights
    if delta >= 0:
        w_pf_tan += w_ltp_tan * sum_pf * sum_tan * delta

        for ii in range(vis_dim**2):
            w_vis_msn_d1[
                ii, :] += w_ltp_msn_d1 * sum_vis[ii] * sum_msn_d1 * delta

    else:
        w_pf_tan += w_ltd_tan * sum_pf * sum_tan * delta

        for ii in range(vis_dim**2):
            w_vis_msn_d1[
                ii, :] += w_ltd_msn_d1 * sum_vis[ii] * sum_msn_d1 * delta

    w_pf_tan = np.clip(w_pf_tan, 0.01, 1)
    w_vis_msn_d1 = np.clip(w_vis_msn_d1, 0.01, 1)


def reset_network():
    global resp, r, delta
    resp = -1
    r = 0
    delta = 0


def record_trial(i, j):
    resp_rec[i, j] = resp == cat
    r_rec[i, j] = r
    delta_rec[i, j] = delta
    pr_rec[i, j] = pr


#################
# Simulate the model
#################
for i in range(n_simulations):

    print(i)

    stimuli = gen_cat_2()

    for j in range(n_trials):

        cat = stimuli['cat'][j]
        x = stimuli['x'][j]
        y = stimuli['y'][j]

        update_pf()
        update_vis(x, y)

        for k in range(n_steps - 1):

            update_da(k)
            update_tan(k)
            update_msn_d1(k)
            update_gpi(k)
            update_th(k)
            update_pm(k)
            update_response(k)
            update_reward()

        update_weights()

        # if j % 10 == 0:

        #     print(j)
        plot_network()

        record_trial(i, j)
        reset_network()

# plot_learning()
