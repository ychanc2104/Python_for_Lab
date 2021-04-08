from EM_Algorithm.gen_gauss import gen_gauss
from EM_Algorithm.gen_poisson import gen_poisson
import numpy as np

def gen_Poi_step(stepsize=5, tau=1, n_events=30, noise=4, fs=100):
    step = gen_gauss([stepsize], [0.1], [n_events])
    taus = gen_poisson([tau], [n_events])
    signal = []
    counts = 0
    for i in range(len(step)):
        signal += [counts] * int(taus[i] * fs)
        counts += step[i]
    N = gen_gauss([0], [noise], [len(signal)])
    signal = np.array(signal) + N
    return signal

