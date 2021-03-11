from gen_poisson import gen_poisson
import numpy as np
from EM_stepsize import collect_EM_results, cal_improvement
import matplotlib.pyplot as plt


def exp_pdf(t, tau):
    y = tau*np.exp(-t/tau)
    return y

##  calculate the probability belonging to each cluster, (m,s)
def weighting(data, tau, f):
    n_components = len(tau)
    # f = np.ones(n_components)/n_components
    n_sample = len(data)
    likelihood = np.zeros((n_components, n_sample))
    for i, x in enumerate(data):
        p = exp_pdf(x, tau)
        likelihood[:, i] = f * p / sum(f * p)
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
    data = gen_poisson(tau=[1], n_sample=[n_sample])

    # likelihood = weighting(data, tau, f)
    # tau2, f2 = update_tau_f(likelihood, data)
    tau = [2]
    n_components = 1
    improvement = 10
    tolerance = 10e-5
    j = 0
    f = np.ones(n_components) / n_components
    tau_save = []
    f_save = []
    while (j < 100 or improvement > tolerance) and j < 5000:
        tau_save, f_save = collect_EM_results([tau_save, tau], [f_save, f])
        likelihood = weighting(data, tau, f)
        tau, f = update_tau_f(likelihood, data)
        improvement = cal_improvement([tau_save[-n_components:], tau], [f_save[-n_components:], f])
        j += 1

    ##  plot EM results(mean, std, fractio with iteration)
    plt.plot(np.arange(0, j), tau_save)
    plt.plot(np.arange(0, j), f_save)


    # bin_width = (12/n_sample)**(1/3)*np.mean(data) ## scott's formula for poisson process
    # bin_number = int((max(data)-min(data))/bin_width)
    # pd, center = binning(data, bin_number, xlabel='dwell time (s)')