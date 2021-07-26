
from OT.PSD import OT_PSD
from OT.gen_Poisson_step import gen_Poi_step, gen_Poi_2step
from basic.fitting import linear_eq, L_fit

import matplotlib.pyplot as plt
import numpy as np
import random
import statistics as stat
from basic.filter import MA

from matplotlib import rcParams
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial"]
rcParams.update({'font.size': 12})




class VA:
    def __init__(self, signal, fs, t_end=1):
        self.signal = signal
        self.fs = fs
        self.t_end = t_end

    def analyze(self):
        signal = self.signal
        fs = self.fs
        t_end = self.t_end
        varX, semX, xm = [], [], []
        ### calculate variance
        t = np.arange(0.05, (t_end - 0.05) / 2, 2/fs)
        for ti in t:
            n_interval = int(np.floor(ti * fs))  ## time diff for calculating variance
            n_row = int(np.floor(len(signal) / n_interval))  ##
            n_samples = n_interval * n_row
            x = signal[0:n_samples]
            x_reshape = x.reshape(n_row, n_interval)
            x_diff = np.diff(x_reshape, axis=0).ravel()
            varX += [stat.variance(x_diff)]  ## sample variance
            semX += [np.std(varX, ddof=1)]
            xm += [np.mean(x_diff)]
        return t, varX, semX


# stepsize = np.array([5, 10])
# tau = np.array([1, 2])
# fs = 100 ## Sampling frequency (Hz)
# n_events = np.array([100,200])

stepsize = np.array([5, 10])
tau = np.array([0.5, 1])
s = np.array([1, 2])
fs = 100 ## Sampling frequency (Hz)
n_events = np.array([100, 50])
noise = 1
figsize = (6,5)
fontsize = 12

signal_connect = gen_Poi_2step(stepsize=stepsize, s=s, tau=tau, n_events=n_events, noise=noise, fs=fs)
t = np.arange(len(signal_connect))/fs
fig, ax = plt.subplots(figsize=figsize)
ax.plot(t, signal_connect)
ax.set_xlabel('Time (s)', fontsize=fontsize)
ax.set_ylabel('Signal (a.u.)', fontsize=fontsize)
ax.set_xlim(0,10)
ax.set_ylim(0,100)
ax.spines[:].set_linewidth('1.5') ## xy, axis width
ax.tick_params(width=1.5) ## tick width

VA_test = VA(signal_connect, fs, t_end=2)
time, varX, semX = VA_test.analyze()
slope, intercept = L_fit(time, varX)
slope_e = np.sum(stepsize**2/tau*n_events/sum(n_events))

fig, ax = plt.subplots(figsize=figsize)
ax.errorbar(time, varX, yerr=semX, color='dodgerblue', marker='o', ls='--', capsize=5, capthick=1, ecolor='black')
ax.plot(time, linear_eq(time, slope_e, 2), 'r--')
ax.set_xlabel('Time (s)', fontsize=fontsize)
ax.set_ylabel('Variance ($\mathregular{count^2}$)', fontsize=fontsize)
ax.spines[:].set_linewidth('1.5')
ax.tick_params(width=1.5)