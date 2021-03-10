

import numpy as np
import matplotlib.pyplot as plt
import math
import random

def gen_gauss(mean, std, n_sample):
    data = []
    for i in range(n_sample):
        data = np.append(data, random.gauss(mean, std))
    return data


if __name__ == "__main__":
    mean = 5
    std = 5
    n_sample = 200
    bin_number = np.log2(n_sample).astype('int') + 1
    data = gen_gauss(mean, std, n_sample)

    count, edges = np.histogram(data, bin_number)
    center = []
    edges = list(edges)
    for i in range(len(edges) - 1):
        center += [(edges[i] + edges[i + 1]) / 2]

    plt.figure()
    plt.hist(data, bin_number,density=True, histtype="bar",color="lightblue", edgecolor="white")
