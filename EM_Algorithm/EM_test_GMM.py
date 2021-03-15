
from matplotlib import rcParams
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial"]
from EM_Algorithm.gen_gauss import gen_gauss
from basic.binning import binning
import numpy as np
import matplotlib.pyplot as plt
import math

##  args: list of parameters, x: np array
def oneD_gaussian(x, args):
    # x: (n,)
    # args: (k), args[0,1,2...,k-1]: (m)
    # output: (k,n)
    f = np.array(args[0])
    xm = np.array(args[1])
    s = np.array(args[2])
    y = []
    for f, xm, s in zip(f, xm, s):
        # y += [f*1/s/np.sqrt(2*math.pi)*np.exp(-(x-xm)**2/2/s**2)]
        if s <= 1.0:
            s = 1.0
        y += [f*1/s/np.sqrt(2*math.pi)*np.exp(-(x-xm)**2/2/s**2)]
    return np.array(y)

def log_oneD_gaussian(x, args):
    f = np.array(args[0])
    xm = np.array(args[1])
    s = np.array(args[2])
    lny = []
    for f, xm, s in zip(f, xm, s):
        if s <= 1.0:
            s = 1.0
        lny += [np.log(f) - np.log(s*np.sqrt(2*math.pi)) - (x-xm)**2/2/s**2]
    return np.array(lny)

##  plot data histogram and its gaussian EM (GMM) results
def plot_fit_gauss(data, f, m, s):
    n_components = len(m)
    bin_number = np.log2(len(data)).astype('int') + 8
    pd, center = binning(data, bin_number) # plot histogram
    data_std_new = np.std(data, ddof=1)
    x = np.arange(0, max(data)+data_std_new, 0.05)
    y_fit = oneD_gaussian(x, [f, m, s])
    for i in range(n_components):
        plt.plot(x, y_fit[i,:], '-')
    plt.plot(x, sum(y_fit), '-')
    plt.xlabel('step size (nm)', fontsize=15)
    plt.ylabel('probability density (1/$\mathregular{nm^2}$)', fontsize=15)

def GMM(data, n_components, tolerance=10e-5):
    ##  initialize EM parameters
    m, s, f = init_GMM(data, n_components=n_components)
    j = 0
    m_save = []
    s_save = []
    f_save = []
    improvement = 10
    while (j < 100 or improvement > tolerance) and j < 5000:
        m_save, s_save, f_save = collect_EM_results([m_save, m], [s_save, s], [f_save, f])
        prior_prob = weighting(data, m, s, f)
        m, s, f = update_m_s_f(prior_prob, data)
        improvement = cal_improvement([m_save[-n_components:], m], [s_save[-n_components:], s], [f_save[-n_components:], f])
        j += 1
    m_save, s_save, f_save = collect_EM_results([m_save, m], [s_save, s], [f_save, f])
    m_save, s_save, f_save = reshape_all(m_save, s_save, f_save, n_components=n_components)
    label = get_label(data, prior_prob)
    data_cluster = [data[label == i] for i in range(n_components)]
    ln_likelihood = [sum(log_oneD_gaussian(data, args=[f, m, s])[i]) for i,data in enumerate(data_cluster)]
    ln_likelihood = sum(ln_likelihood)
    BIC = -2*ln_likelihood + (n_components*3-1)*np.log(len(data))
    AIC = -2*ln_likelihood + (n_components*3-1)*2
    return f_save, m_save, s_save, label, data_cluster, BIC, AIC

    ##  initialize mean, std and fraction of gaussian
def init_GMM(data, n_components):
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    m = np.arange(mean-1.5*std, mean+1.5*std+1, 3*std/(n_components-1))
    s = np.array([std]*n_components)
    f = np.ones(n_components) / n_components
    return m, s, f

##  calculate the probability belonging to each cluster, (m,s)
def weighting(data, m, s, f):
    p = oneD_gaussian(data, args=[f, m, s])
    prior_prob = p / sum(p)
    return prior_prob

##  update mean, std and fraction using matrix multiplication, (n_feture, n_sample) * (n_sample, 1) = (n_feture, 1)
def update_m_s_f(likelihood, data):
    n_sample = len(data)
    m = np.matmul(likelihood, data) / np.sum(likelihood, axis=1)
    s = np.sqrt(np.matmul(likelihood, data ** 2)/np.sum(likelihood, axis=1)-m**2)
    f = np.sum(likelihood, axis=1) / n_sample
    return m, s, f

def plot_EM_result(result, ax, xlabel='iteration', ylabel='value'):
    n_feature = result.shape[1]
    iteration = result.shape[0]
    for i in range(n_feature):
        ax.plot(np.arange(0, iteration), result[:, i], '-o')
    ax.set_ylabel(f'{ylabel}', fontsize=15)

def plot_EM_results(m, s, f):
    fig, axs = plt.subplots(3, sharex=True)
    axs[2].set_xlabel('iteration', fontsize=15)
    plot_EM_result(m, axs[0], ylabel='mean')
    plot_EM_result(s, axs[1], ylabel='std')
    plot_EM_result(f, axs[2], ylabel='fraction')

def cal_improvement(*args):
    ##  arg: [m_old, m_new]
    improvement = 0
    for arg in args:
        dx = abs(arg[0] - arg[1])
        improvement = max(np.append(improvement, dx)) # take max of all args diff
    return improvement

def collect_EM_results(*args):
    results = []
    for arg in args:
        results += [np.append(arg[0], arg[1])]
    return results

def get_label(data, likelihood):
    label = []
    for i, x in enumerate(data):
        p = likelihood[:, i]
        label += [np.argmax(p)]
    label = np.array(label)
    return label

def reshape_all(*args, n_components):
    results = []
    n_sample = int(len(args[0])/n_components)
    for arg in args:
        results += [np.reshape(arg, (n_sample, n_components))]
    return results

if __name__ == '__main__':
    n_sample = 200
    data = gen_gauss(mean=[5,10], std=[2,2], n_sample=[n_sample]*2)

    ##  fit GMM
    n_components = 2
    f_save, m_save, s_save, label, data_cluster, BIC, AIC = GMM(data, n_components, tolerance=10e-5)
    m = m_save[-1, :]
    s = s_save[-1, :]
    f = f_save[-1, :]

    ##  plot EM results(mean, std, fractio with iteration)
    plot_EM_results(m_save, s_save, f_save)

    ##  plot data histogram and its gaussian EM (GMM) results
    plot_fit_gauss(data, f, m, s)

