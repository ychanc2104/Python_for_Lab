from OT.PSD import OT_PSD
from OT.gen_Poisson_step import gen_Poi_step
import matplotlib.pyplot as plt
import numpy as np
import random
from basic.filter import MA


stepsize = 10
tau = 1
# n_events = 30
# noise = 1
fs = 100 ## Sampling frequency (Hz)

Fs_spatial = 1
F_resolution = 0.0001
#
# freq_all = []
# psd_all = []
freq_all_c = []
psd_all_c = []
t_AFC_all = []
AFC_all = []
signal_connect = []
L_cum=0
for i in range(2):
    n_events = int(random.random()*20+500)
    noise = int(random.random()*1+1)
    signal = gen_Poi_step(stepsize=stepsize, tau=tau, n_events=n_events, noise=noise, fs=fs)
    signal_connect = np.append(signal_connect, signal+L_cum)
    L_cum += np.mean(signal[-20:])


    PSD = OT_PSD(signal, fs=fs, Fs_spatial=Fs_spatial, F_resolution=F_resolution)
    t_AFC_conn, AFC_conn = PSD.t_ACF, PSD.ACF

    freq, psd = PSD.get_PSD()
    # freq_all += [freq]
    # psd_all += [psd]
    freq_all_c = np.append(freq_all_c, freq)
    psd_all_c = np.append(psd_all_c, psd)
    t_AFC_all += [t_AFC_conn[:500]]
    AFC_all += [AFC_conn[:500]]

PSD_connect = OT_PSD(signal_connect, fs=fs, Fs_spatial=Fs_spatial, F_resolution=F_resolution)
freq_conn, psd_conn = PSD_connect.get_PSD()

t_AFC = np.mean(np.array(t_AFC_all), axis=0)
AFC = np.mean(np.array(AFC_all), axis=0)

# psd = np.mean(np.array(psd_all), axis=0)
# freq = np.mean(np.array(freq_all), axis=0)
# fig, ax = plt.subplots(figsize=(10,8))
# ax.plot(freq, psd, '--o')
# ax.set_xlim(0, Fs_spatial/2)
# ax.set_ylim(0, psd[np.argsort(psd)[-2]]*2)
# ax.set_xlabel('spatial frequency (1/count)')
# ax.set_ylabel('PSD')

fig, ax = plt.subplots(figsize=(10,8))
# t_AFC = np.sort(t_AFC_all)
# AFC = psd_all_c[np.argsort(t_AFC_all)]
ax.plot(t_AFC, AFC, '-')
ax.set_xlim(0, 20)
# ax.set_xlim(0, 100)
# ax.set_ylim(0.000, 0.1)
ax.set_xlabel('Distance (count)')
ax.set_ylabel('Autocorrelation')



fig, ax = plt.subplots(figsize=(10,8))
ax.plot(freq_all_c, psd_all_c, '.')
ax.set_xlim(0, Fs_spatial/2)
ax.set_ylim(0, psd[np.argsort(psd)[-2]]*2)
ax.set_xlabel('spatial frequency (1/count)')
ax.set_ylabel('PSD')

# fig, ax = plt.subplots(figsize=(10,8))
# ax.plot(freq_conn, psd_conn, '-.')
# ax.set_xlim(0, Fs_spatial/2)
# ax.set_ylim(0, psd[np.argsort(psd)[-2]]/20)
# ax.set_xlabel('spatial frequency (1/count)')
# ax.set_ylabel('PSD')