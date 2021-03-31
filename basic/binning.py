import numpy as np
import matplotlib.pyplot as plt

def binning(data, bin_number, xlabel='value', ylabel='probability density'):
    count, edges = np.histogram(data, bin_number)
    center = []
    edges = list(edges)
    for i in range(len(edges) - 1):
        center += [(edges[i] + edges[i + 1]) / 2]
    binsize = center[1] - center[0]
    pd = count/sum(count)/binsize
    fig, ax = plt.subplots(figsize=(10,8))
    ax.bar(center, pd, width=binsize, color="silver", edgecolor="white")
    ax.set_xlabel(f'{xlabel}', fontsize=15)
    ax.set_ylabel(f'{ylabel}', fontsize=15)
    return pd, center, fig, ax