from gen_poisson import gen_poisson
import numpy as np
from EM_stepsize import collect_EM_results, cal_improvement
import matplotlib.pyplot as plt
from binning import binning



def exp_pdf(t, tau):
    y = 1/tau*np.exp(-t/tau)
    return y

def ln_exp_pdf(t, tau):
    lny = -np.log(tau) - t/tau
    return lny

##  calculate the probability belonging to each cluster, (m,s)
def weighting(data, tau, f):
    n_components = len(tau)
    # f = np.ones(n_components)/n_components
    n_sample = len(data)
    likelihood = np.zeros((n_components, n_sample))
    for i, x in enumerate(data):
        p = ln_exp_pdf(x, tau)
        likelihood[:, i] = f * np.exp(p) / sum(f * np.exp(p))
    return likelihood

##  update mean, std and fraction using matrix multiplication, (n_feture, n_sample) * (n_sample, 1) = (n_feture, 1)
def update_tau_f(likelihood, data):
    n_sample = len(data)
    tau = np.matmul(likelihood, data) / np.sum(likelihood, axis=1)
    f = np.sum(likelihood, axis=1) / n_sample
    return tau, f

def plot_EM_results(tau, f):
    fig, axs = plt.subplots(2, sharex=True)
    axs[1].set_xlabel('iteration', fontsize=15)
    plot_EM_result(tau, axs[0], ylabel='mean')
    plot_EM_result(f, axs[1], ylabel='fraction')

def plot_EM_result(result, ax, xlabel='iteration', ylabel='value'):
    n_feature = 1
    iteration = result.shape[0]
    for i in range(n_feature):
        ax.plot(np.arange(0, iteration), result[:, i], '-o')
    ax.set_ylabel(f'{ylabel}', fontsize=15)

if __name__ == "__main__":
    n_sample = 500
    data = gen_poisson(tau=[0.1, 2, 10], n_sample=[200, 200, 2000])
    n_sample = len(data)
    # likelihood = weighting(data, tau, f)
    # tau2, f2 = update_tau_f(likelihood, data)
    tau = [0.4, 10, 15]
    # tau= [5]
    n_components = len(tau)
    improvement = 10
    tolerance = 10e-5
    j = 0
    f = np.ones(n_components) / n_components
    tau_save = []
    f_save = []
    while (j < 10 or improvement > tolerance) and j < 5000:
        tau_save, f_save = collect_EM_results([tau_save, tau], [f_save, f])
        likelihood = weighting(data, tau, f)
        tau, f = update_tau_f(likelihood, data)
        improvement = cal_improvement([tau_save[-n_components:], tau], [f_save[-n_components:], f])
        j += 1
    tau_save = np.reshape(tau_save, (j ,n_components))
    f_save = np.reshape(f_save, (j ,n_components))
    ##  plot EM results(mean, std, fractio with iteration)
    plt.plot(np.arange(0, j), tau_save, 'o-')
    plt.figure()
    plt.plot(np.arange(0, j), f_save, 'o-')

    label = []
    for i, x in enumerate(data):
        p = likelihood[:, i]
        label += [np.argmax(p)]
    label = np.array(label)
    data_cluster = [data[label == i] for i in range(n_components)]

    bin_width = (12/n_sample)**(1/3)*np.mean(data)/n_components**1.5 ## scott's formula for poisson process
    bin_number = int((max(data)-min(data))/bin_width)
    pd, center = binning(data, bin_number, xlabel='dwell time (s)')
    
    x = np.arange(0.1, max(data), 0.01)
    y_fit = np.array([f[i]*exp_pdf(x, tau[i]) for i in range(n_components)])
    for i in range(n_components):
        plt.plot(x, y_fit[i,:], '-')
    plt.plot(x, sum(y_fit), '-')
    plt.xlabel('dwell time (s)', fontsize=15)
    plt.ylabel('probability density (1/$\mathregular{s^2}$)', fontsize=15)
    plt.xlim([0,np.mean(data)*2])
    