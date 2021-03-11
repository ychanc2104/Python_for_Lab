import numpy as np
import random
from binning import binning

def gen_poisson(tau, n_sample):
    data = []
    for tau,n in zip(tau, n_sample):
        for i in range(n):
            data = np.append(data, random.expovariate(1/tau))
    return data

if __name__ == "__main__":
    n_sample = 500
    data = gen_poisson(tau=[2], n_sample=[n_sample])
    bin_width = (12/n_sample)**(1/3)*np.mean(data) ## scott's formula for poisson process
    bin_number = int((max(data)-min(data))/bin_width)
    pd, center = binning(data, bin_number, xlabel='dwell time (s)')