import numpy as np
import random
from binning import binning

def gen_gauss(mean, std, n_sample):
    data = []
    for m,s,n in zip(mean,std,n_sample):
        for i in range(n):
            data = np.append(data, random.gauss(m, s))
    return data

if __name__ == "__main__":
    mean = 5
    std = 3
    n_sample = 200
    data = gen_gauss(mean=[5, 15, 25], std=[2, 2, 2], n_sample=[n_sample,n_sample,n_sample])
    bin_number = np.log2(len(data)).astype('int') + 1
    pd, center = binning(data, bin_number)

