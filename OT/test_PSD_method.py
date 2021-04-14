from OT.PSD import OT_PSD
from OT.gen_Poisson_step import gen_Poi_step
import matplotlib.pyplot as plt
import numpy as np
import random

stepsize = 4
tau = 1
# n_events = 30
# noise = 1
fs = 100 ## Sampling frequency (Hz)

Fs_spatial = 1
F_resolution = 0.001

freq_all = []
psd_all = []
freq_all_c = []
psd_all_c = []
for i in range(10):
    n_events = int(random.random()*20+500)
    noise = int(random.random()*1+1)
    signal = gen_Poi_step(stepsize=stepsize, tau=tau, n_events=n_events, noise=noise, fs=fs)
    PSD = OT_PSD(signal, fs=fs, Fs_spatial=Fs_spatial, F_resolution=F_resolution)
    freq, psd = PSD.get_PSD()
    # freq_all += [freq]
    # psd_all += [psd]
    freq_all_c = np.append(freq_all_c, freq)
    psd_all_c = np.append(psd_all_c, psd)


# psd = np.mean(np.array(psd_all), axis=0)
# freq = np.mean(np.array(freq_all), axis=0)
# fig, ax = plt.subplots(figsize=(10,8))
# ax.plot(freq, psd, '--o')
# ax.set_xlim(0, Fs_spatial/2)
# ax.set_ylim(0, psd[np.argsort(psd)[-2]]*2)
# ax.set_xlabel('spatial frequency (1/count)')
# ax.set_ylabel('PSD')

fig, ax = plt.subplots(figsize=(10,8))
ax.plot(freq_all_c, psd_all_c, '.')
ax.set_xlim(0, Fs_spatial/2)
ax.set_ylim(0, psd[np.argsort(psd)[-2]]*2)
ax.set_xlabel('spatial frequency (1/count)')
ax.set_ylabel('PSD')

