
from EM_Algorithm.gen_gauss import gen_gauss
from EM_Algorithm.gen_poisson import gen_poisson
import matplotlib.pyplot as plt
from basic.binning import binning2
import numpy as np
import math

class OT_PSD:
    def __init__(self, signal, fs, Fs_spatial=2, F_resolution=0.01):
        self.signal = signal
        self.fs = fs
        self.Fs_spatial = Fs_spatial
        self.F_resolution = F_resolution
        self.t_ACF, self.ACF = self.get_auto_corr()

    def get_auto_corr(self):
        signal = self.signal
        Fs_spatial = self.Fs_spatial
        F_resolution = self.F_resolution
        pd, center, fig, ax = binning2(signal, 1/Fs_spatial, show=False)
        auto_correlation = np.correlate(pd, pd, 'full')
        index = np.argmax(auto_correlation)
        auto_correlation_R = auto_correlation[index:index + int(Fs_spatial / F_resolution)] ##  picking right hand side
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

