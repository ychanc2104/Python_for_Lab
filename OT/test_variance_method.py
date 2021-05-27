from OT.PSD import OT_PSD
from OT.gen_Poisson_step import gen_Poi_step, gen_Poi_2step
from basic.fitting import linear_eq, L_fit

import matplotlib.pyplot as plt
import numpy as np
import random
import statistics as stat
from basic.filter import MA

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
        t = np.arange(0.05, (t_end - 0.05) / 2, 10/fs)
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


stepsize = np.array([5, 5])
tau = np.array([1, 1])
fs = 100 ## Sampling frequency (Hz)
n_events = [100,100]
noise = 1

signal_connect = gen_Poi_2step(stepsize=stepsize, tau=tau, n_events=n_events, noise=noise, fs=fs)
t = np.arange(len(signal_connect))/fs
fig, ax = plt.subplots()
ax.plot(t, signal_connect)
ax.set_xlabel('Time (s)', fontsize=22)
ax.set_ylabel('Signal (a.u.)', fontsize=22)


VA_test = VA(signal_connect, fs, t_end=10)
time, varX, semX = VA_test.analyze()
slope, intercept = L_fit(time, varX)
slope_e = np.sum(stepsize**2/tau)

fig, ax = plt.subplots(figsize=(10,8))
ax.errorbar(time, varX, yerr=semX, color='dodgerblue', marker='o', ls='--', capsize=5, capthick=1, ecolor='black')
ax.plot(time, linear_eq(time, 25, 1), 'r--')
ax.set_xlabel('Time (s)', fontsize=22)
ax.set_ylabel('Variance ($\mathregular{count^2}$)', fontsize=22)