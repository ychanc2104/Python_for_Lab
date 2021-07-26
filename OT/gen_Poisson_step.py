import random

from EM_Algorithm.gen_gauss import gen_gauss
from EM_Algorithm.gen_poisson import gen_poisson
import numpy as np
import matplotlib.pyplot as plt


def gen_Poi_step(stepsize=5, s=2, tau=1, n_events=30, noise=4, fs=100):
    step = gen_gauss([stepsize], [s], [n_events])
    taus = gen_poisson([tau], [n_events])
    signal = []
    counts = 0
    for i in range(len(step)):
        signal += [counts] * int(taus[i] * fs)
        counts += step[i]
    N = gen_gauss([0], [noise], [len(signal)])
    signal = np.array(signal) + N
    return signal

def gen_Poi_2step(stepsize=[5,10], s=[1,1], tau=[1,4], n_events=[30,30], noise=1, fs=100):
    index = np.arange(sum(n_events))
    random.shuffle(index)
    step = gen_gauss(stepsize, s, n_events)[index]
    taus = gen_poisson(tau, n_events)[index]
    signal = []
    counts = 0
    for i in range(len(step)):
        signal += [counts] * int(taus[i] * fs)
        counts += step[i]
    N = gen_gauss([0], [noise], [len(signal)])
    signal = np.array(signal) + N
    return signal

if __name__ == '__main__':
    stepsize = 8
    tau = 1
    n_events = 8
    noise = 2
    fs = 100
    signal = gen_Poi_step(stepsize=stepsize, tau=tau, n_events=n_events, noise=noise, fs=fs)
    time = np.arange(len(signal))/fs
    fig, ax = plt.subplots(figsize=(10,8))
    ax.plot(time, signal)
    ax.set_xlabel('time (s)', fontsize=16)
    ax.set_ylabel('signal (a.u.)', fontsize=16)
