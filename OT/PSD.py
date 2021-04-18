
from EM_Algorithm.gen_gauss import gen_gauss
from EM_Algorithm.gen_poisson import gen_poisson
import matplotlib.pyplot as plt
from basic.binning import binning, binning2
import numpy as np
import math

class OT_PSD:
    def __init__(self, signal, fs, Fs_spatial=2, F_resolution=0.01, bintype='set_width', bin_number=10):
        self.signal = signal
        self.fs = fs
        self.Fs_spatial = Fs_spatial
        self.F_resolution = F_resolution
        self.t_ACF, self.ACF = self.get_auto_corr(bintype=bintype, bin_number=bin_number)

    def get_auto_corr(self, bintype='set_width', bin_number=10, density=False):
        signal = self.signal
        F_resolution = self.F_resolution
        if bintype == 'set_width':
            Fs_spatial = self.Fs_spatial
            pd, center, fig, ax = binning2(signal, binwidth=1/Fs_spatial, xlabel='count', show=False, density=density)
        else:
            pd, center, fig, ax = binning(signal, bin_number, xlabel='count', ylabel='probability density', show=False, density=density)
            Fs_spatial = 1/abs(center[1]-center[0])
            self.Fs_spatial = Fs_spatial

        auto_correlation = np.correlate(pd, pd, 'full')
        # index = np.argmax(auto_correlation)
        index = int(len(auto_correlation)/2)
        auto_correlation_R = auto_correlation[index:] ##  picking right hand side
        t_spatial = np.arange(0, len(auto_correlation_R)) / Fs_spatial
        self.auto_correlation = auto_correlation_R
        self.t_ACF = t_spatial
        return t_spatial, auto_correlation_R

    def get_PSD(self):
        ACF = self.auto_correlation
        Fs_spatial = self.Fs_spatial
        pre_psd = np.fft.fft(ACF)
        psd = abs(pre_psd)
        freq = np.linspace(0, Fs_spatial, len(ACF))
        cut = int(len(psd)/2)
        self.freq = freq[:cut]
        self.psd = psd[:cut]
        return freq, psd

