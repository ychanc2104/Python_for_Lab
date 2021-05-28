from matplotlib import rcParams
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial"]
rcParams.update({'font.size': 18})
import numpy as np
import matplotlib.pyplot as plt

def binning(data, bin_number, xlabel='value', ylabel='probability density',
            show=True, density=True, figsize=(10,8), color="silver", fontsize=22):
    count, edges = np.histogram(data, bin_number)
    center = []
    edges = list(edges)
    for i in range(len(edges) - 1):
        center += [(edges[i] + edges[i + 1]) / 2]
    binsize = center[1] - center[0]
    if density == True:
        pd = count/sum(count)/binsize
    else:
        pd = count
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(center, pd, width=binsize, color=color, edgecolor="white")
    ax.set_xlabel(f'{xlabel}', fontsize=fontsize)
    ax.set_ylabel(f'{ylabel}', fontsize=fontsize)
    if show==False:
        plt.close(fig)
    return pd, center, fig, ax

def binning2(data, binwidth, start=None, end=None, xlabel='value', ylabel='probability density', show=True, density=True):
    if start==None:
        start = min(data)
    if end == None:
        end = max(data)
    bin_edge = np.arange(start, end+1, binwidth)
    center = np.arange(start+binwidth/2, end-binwidth/2+1, binwidth)
    fig, ax = plt.subplots(figsize=(10, 8))
    pd, edges, patches = plt.hist(data, bins=bin_edge, density=density)
    ax.bar(center, pd, width=binwidth, color="silver", edgecolor="white")
    ax.set_xlabel(f'{xlabel}', fontsize=22)
    ax.set_ylabel(f'{ylabel}', fontsize=22)
    if show==False:
        plt.close(fig)
    return pd, center, fig, ax
