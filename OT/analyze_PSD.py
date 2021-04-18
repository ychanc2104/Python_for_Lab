from OT.PSD import OT_PSD
from basic.select import select_file
from basic.filter import MA
from basic.binning import binning2

from matplotlib import rcParams
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial"]
rcParams.update({'font.size': 18})
import pandas as pd

import numpy as np
import random
import math
import statistics as stat
import matplotlib.pyplot as plt
from basic.fitting import linear_eq, L_fit
from basic.file_io import save_img

##  generate #n uniform RV range from 0 to max
def get_uRV(max, n):
    RVs = np.array([int(math.floor(random.uniform(0, max))) for i in range(n)])
    return RVs

def get_boot_select(signals):
    n = len(signals)
    signals_boot = []
    index = get_uRV(n, n)
    for i in index:
        signals_boot += [list(signals)[i]]
    return signals_boot

def get_excel_data(data):
    n_traces = int(data.shape[1] / 4)
    signals = []
    dt = []
    t_end = []  # ending time
    Fs = []
    for i in range(n_traces):
        ##  select and remove nan data
        signal = data[:, 1 + 4 * i]
        signals += [signal[~np.isnan(signal)]]
        dt += [data[1, 0 + 4 * i] - data[0, 0 + 4 * i]]
        t_end += [dt[-1] * len(signals[-1])]
        Fs += [1 / dt[-1]]
    return signals, Fs

def connect_traces(signals):
    signal_connect = []
    L_cum = 0
    ##  for connecting all traces
    for fs,signal in zip(Fs,signals):
        signal_connect = np.append(signal_connect, signal+L_cum)
        L_cum += np.mean(signal[-20:])
    return signal_connect


### import data
# path = select_file()
path = r'/home/hwligroup/Desktop/20210330/YYH_m51_data/time trace/m51 all traces/m51_3.0uM_All.xlsx'
## x:1.8,
df = pd.read_excel(path)
data = np.array(df.dropna(axis='columns', how='all'))
signals, Fs = get_excel_data(data)


freq_all_c = []
psd_all_c = []
t_AFC_all = []
AFC_all = []
n_boot = 2
for i in range(n_boot):
    Fs_spatial = 5
    F_resolution = 0.002

    signals_boot = get_boot_select(signals)
    signal_connect = connect_traces(signals_boot)

    PSD_connect = OT_PSD(signal_connect, fs=np.mean(Fs), Fs_spatial=Fs_spatial, F_resolution=F_resolution, bintype='set_width')
    t_AFC_conn, AFC_conn = PSD_connect.t_ACF, PSD_connect.ACF
    freq_conn, psd_conn = PSD_connect.get_PSD()

    freq_all_c = np.append(freq_all_c, freq_conn)
    psd_all_c = np.append(psd_all_c, psd_conn)
    t_AFC_all += [t_AFC_conn[:200]]
    AFC_all += [AFC_conn[:200]]

t_AFC = np.mean(np.array(t_AFC_all), axis=0)
AFC = np.mean(np.array(AFC_all), axis=0)

# fig, ax = plt.subplots(figsize=(10,8))
# ax.plot(freq_conn, psd_conn, '.')
# ax.set_xlim(0, Fs_spatial/2)
# ax.set_ylim(0, psd[np.argsort(psd)[-2]]*2)
# ax.set_xlabel('spatial frequency (1/count)')
# ax.set_ylabel('PSD')

fig, ax = plt.subplots(figsize=(10,8))
ax.plot(t_AFC, AFC, '-.')
ax.set_xlim(0, 40)
# ax.set_xlim(0, 100)
# ax.set_ylim(0.000, 0.1)
ax.set_xlabel('Distance (count)')
ax.set_ylabel('Autocorrelation')

fig, ax = plt.subplots(figsize=(10,8))
f = np.sort(freq_all_c)
p = psd_all_c[np.argsort(freq_all_c)]
# ax.plot(MA(f,100,mode='silding'), MA(p,100,mode='silding'), '.')
ax.plot(f, p, '-')
ax.set_xlim(0, 0.5)
# ax.set_xlim(0, 100)
# ax.set_ylim(0.000, 0.1)
ax.set_xlabel('Spatial frequency (1/count)')
ax.set_ylabel('Power spectral density(a.u.)')




plt.figure()
plt.plot(signal_connect)