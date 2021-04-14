from OT.PSD import OT_PSD

from matplotlib import rcParams
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial"]
rcParams.update({'font.size': 18})
import pandas as pd
from basic.select import select_file
import numpy as np

import statistics as stat
import matplotlib.pyplot as plt
from basic.fitting import linear_eq, L_fit
from basic.file_io import save_img

### import data
path = select_file()
df = pd.read_excel(path)
data = np.array(df.dropna(axis='columns', how='all'))
n_traces = int(data.shape[1]/4)
signals = []
dt = []
t_end = [] # ending time
Fs = []
freq_all_c = []
psd_all_c = []
Fs_spatial = 1
F_resolution = 0.01
for i in range(n_traces):
    ##  select and remove nan data
    signal = data[:, 1+4*i]
    signals += [signal[~np.isnan(signal)]]
    dt += [data[1, 0+4*i] - data[0, 0+4*i]]
    t_end += [dt[-1]*len(signals[-1])]
    Fs += [1/dt[-1]]

    PSD = OT_PSD(signal, fs=Fs[-1], Fs_spatial=Fs_spatial, F_resolution=F_resolution)
    freq, psd = PSD.get_PSD()
    # freq_all += [freq]
    # psd_all += [psd]
    freq_all_c = np.append(freq_all_c, freq)
    psd_all_c = np.append(psd_all_c, psd)

fig, ax = plt.subplots(figsize=(10,8))
ax.plot(freq_all_c, psd_all_c, '.')
ax.set_xlim(0, Fs_spatial/2)
# ax.set_ylim(0, psd[np.argsort(psd)[-2]]*2)
ax.set_xlabel('spatial frequency (1/count)')
ax.set_ylabel('PSD')