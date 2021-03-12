from EM_Algorithm.gen_poisson import gen_poisson
from EM_Algorithm.EM_stepsize import collect_EM_results, cal_improvement
from basic.binning import binning
import numpy as np
import matplotlib.pyplot as plt

def exp_EM(data, n_components, tolerance=10e-5):
    f, tau = init(data, n_components=n_components)
    improvement = 10
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
    f_save = np.reshape(f_save, (j, n_components))
    tau_save = np.reshape(tau_save, (j, n_components))
    return f_save, tau_save

##  initialize parameters
def init(data, n_components):
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        f = np.ones(n_components) / n_components
        tau = np.arange(abs(mean - 0.5 * std), mean + 0.5 * std + 1, 1*std/(n_components - 1))
        return f, tau

##  args: list
def exp_pdf(t, args):
    f = np.array(args[0])
    tau = np.array(args[1])
    try:
        n_col = len(t)
    except:
        n_col = 1
    try:
        n_row = len(f)
    except:
        n_row = 1
    y = np.zeros((n_row, n_col))
    for i in range(n_row):
        y[i,:] = f[i] * 1/tau[i]*np.exp(-t/tau[i])
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
        p = exp_pdf(x, [f,tau])
        likelihood[:, i] = (p / sum(p)).ravel()
    return likelihood

##  update mean, std and fraction using matrix multiplication, (n_feture, n_sample) * (n_sample, 1) = (n_feture, 1)
def update_tau_f(likelihood, data):
    n_sample = len(data)
    tau = np.matmul(likelihood, data) / np.sum(likelihood, axis=1)
    f = np.sum(likelihood, axis=1) / n_sample
    return tau, f

def plot_EM_results(results, labels):
    n_components = len(results)
    fig, axs = plt.subplots(n_components, sharex=True)
    axs[-1].set_xlabel('iteration', fontsize=15)
    for i in range(n_components):
        plot_EM_result(results[i], axs[i], ylabel=f'{labels[i]}')

##  result: (iteration, n_components)
def plot_EM_result(result, ax, xlabel='iteration', ylabel='value'):
    iteration = result.shape[0]
    n_components = result.shape[1]
    for i in range(n_components):
        ax.plot(np.arange(0, iteration), result[:, i], '-o')
    ax.set_ylabel(f'{ylabel}', fontsize=15)

##  plot data histogram and its gaussian EM (GMM) results, results:
def plot_fit_pdf(data, function, results):
    n_components = len(results[0])
    bin_width = (12/n_sample)**(1/3)*np.mean(data)/n_components**1.3 ## scott's formula for poisson process
    bin_number = int((max(data)-min(data))/bin_width)
    pd, center = binning(data, bin_number) # plot histogram
    data_std_new = np.std(data, ddof=1)
    x = np.arange(0.1, max(data), 0.01)
    y_fit = function(x, results)
    for i in range(n_components):
        plt.plot(x, y_fit[i,:], '-')
    plt.plot(x, sum(y_fit), '-')
    plt.xlabel('dwell time (s)', fontsize=15)
    plt.ylabel('probability density (1/$\mathregular{s^2}$)', fontsize=15)
    plt.xlim([0, data_std_new*2])


if __name__ == "__main__":
    ##  produce data
    data = gen_poisson(tau=[0.1, 2, 10], n_sample=[200, 200, 200])
    n_sample = len(data)

    ##  fit EM
    n_components = 3
    f_save, tau_save = exp_EM(data, n_components, tolerance=10e-5)

    ##  plot EM results(mean, std, fractio with iteration)
    plot_EM_results([f_save, tau_save], ['fraction', 'dwell time'])

    ##  plot data histogram and its gaussian EM (GMM) results
    plot_fit_pdf(data, exp_pdf, [f_save[-1,:], tau_save[-1,:]])
