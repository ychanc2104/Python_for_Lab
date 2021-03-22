
from matplotlib import rcParams
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial"]
# rcParams.update({'font.size': 18})
from basic.binning import binning
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from basic.file_io import save_img
from lifelines import KaplanMeierFitter
import pandas as pd


### data: (n,1)-array
class EM:
    def __init__(self, data):
        self.data = data.reshape(-1, 1)

    def skGMM(self, n_components, tolerance=10e-5):
        # self.n_components = n_components
        data = self.data
        n_sample = len(data)

        gmm = GaussianMixture(n_components=n_components, tol=tolerance).fit(data)
        labels = gmm.predict(data)
        data_cluster = [data[labels == i] for i in range(n_components)]
        p = gmm.predict_proba(data).T
        f = np.sum(p, axis=1) / n_sample
        m = np.matmul(p, data).ravel() / np.sum(p, axis=1)
        s = np.sqrt(np.matmul(p, data ** 2).ravel() / (np.sum(p, axis=1)) - m ** 2)
        self.f_sk = f.reshape(1, n_components)
        self.m_sk = m.reshape(1, n_components)
        self.s_sk = s.reshape(1, n_components)
        return f, m, s, labels, data_cluster


    def GMM(self, n_components, tolerance=1e-2):
        """EM algorithm with pdf=Gaussian (GMM)
        Parameters
        ----------
        n_components : int
            Number of components.
        tolerance : float
            Convergence criteria
        data : array (n_samples,1)
        Returns
        -------

        """
        ## (f,m,s) are growing array
        data = self.data
        self.tolerance = tolerance
        ##  initialize EM parameters
        f, m, s, loop, improvement = self.__init_GMM(n_components=n_components)
        while (loop < 10 or improvement > tolerance) and loop < 500:
            prior_prob = self.__weighting(f, m, s, function=ln_oneD_gaussian)
            f, m, s = self.__update_f_m_s(prior_prob, f, m, s)
            improvement = self.__cal_improvement(f, m, s)
            loop += 1
        self.__weighting(f, m, s, function=ln_oneD_gaussian) # update prior prob
        f, m, s = self.__reshape_all(f, m, s, n_rows=loop+1, n_cols=n_components)
        self.f = f
        self.m = m
        self.s = s
        labels, data_cluster, ln_likelihood = self.predict(data)
        self.data_cluster = data_cluster
        return f, m, s, labels, data_cluster

    def PEM(self, n_components, tolerance=1e-2):
        data = self.data
        self.tolerance = tolerance

        f, tau, s, loop, improvement = self.__init_PEM(n_components=n_components)
        while (loop < 10 or improvement > tolerance) and loop < 500:
            prior_prob = self.__weighting(f, tau, function=ln_exp_pdf)
            f, tau, s = self.__update_f_m_s(prior_prob, f, tau, s)
            improvement = self.__cal_improvement(f, tau)
            loop += 1
        self.__weighting(f, tau, function=ln_exp_pdf)
        f, tau, s = self.__reshape_all(f, tau, s, n_rows=loop+1, n_cols=n_components)
        self.f = f
        self.m = tau
        self.s = s
        labels, data_cluster, ln_likelihood = self.predict_p(data)
        return f, tau, s, labels, data_cluster

    def opt_components(self, tolerance=1e-2, mode='GMM', criteria='AIC', figure=False):
        ##  find best n_conponents
        data = self.data
        BICs = []
        AICs = []
        BIC_owns = []
        AIC_owns = []
        LLE = []
        n_clusters = np.arange(1, 6)
        for c in n_clusters:
            if mode == 'GMM':
                f, tau, s, labels, data_cluster = self.GMM(n_components=c, tolerance=tolerance)
                gmm = GaussianMixture(n_components=c, tol=tolerance).fit(data)
                BICs += [gmm.bic(data)]
                AICs += [gmm.aic(data)]
            else:
                f, tau, s, labels, data_cluster = self.PEM(n_components=c, tolerance=tolerance)

            BIC_owns += [self.__BIC()]
            AIC_owns += [self.__AIC()]
            LLE += [self.ln_likelihood]
        if figure == True:
            plt.figure()
            plt.plot(n_clusters, BIC_owns, 'o')
            plt.title('BIC')
            plt.xlabel('n_components')
            plt.ylabel('BIC_owns')
            plt.figure()
            plt.plot(n_clusters, AIC_owns, 'o')
            plt.title('AIC')
            plt.xlabel('n_components')
            plt.ylabel('AIC_owns')

        ##  get optimal components
        if criteria=='AIC':
            opt_components = n_clusters[np.argmin(AIC_owns)]
        else:
            opt_components = n_clusters[np.argmin(BIC_owns)]
        self.LLE = LLE
        self.BICs = BICs
        self.AICs = AICs
        self.BIC_owns = BIC_owns
        self.AIC_owns = AIC_owns
        return opt_components


    ##  get predicted data_cluster and its log-likelihood
    def predict(self, data):
        """predict data cluster
        Parameters
        ----------
        ln_likelihood : int
            Number of components.
        tolerance : float
            Convergence criteria
        data : array (n_samples,1)

        prior_prob: array (n_components, n_sample)

        Returns
        -------

        """

        f = self.f[-1, :]
        m = self.m[-1, :]
        s = self.s[-1, :]

        n_components = len(f)
        # prior_prob = self.prior_prob
        p = np.exp(ln_oneD_gaussian(data, args=[f,m,s])) ##(n_components, n_samples)
        prior_prob = p / sum(p)
        labels = np.array([np.argmax(prior_prob[:, i]) for i in range(len(data))])  ## find max of prob
        data_cluster = [data[labels == i] for i in range(n_components)]
        ln_likelihood = sum([np.log(sum(np.exp(ln_oneD_gaussian(data[i], args=[f, m, s]).ravel()))) for i in range(len(data))])

        self.ln_likelihood = ln_likelihood
        return labels, data_cluster, ln_likelihood

    ##  get predicted data_cluster and its log-likelihood
    def predict_p(self, data):
        """predict data cluster
        Parameters
        ----------
        ln_likelihood : int
            Number of components.
        tolerance : float
            Convergence criteria
        data : array (n_samples,1)

        prior_prob: array (n_components, n_sample)

        Returns
        -------

        """

        f = self.f[-1, :]
        m = self.m[-1, :]
        s = self.s[-1, :]

        n_components = len(f)
        p = np.exp(ln_exp_pdf(data, args=[f,m])) ##(n_components, n_samples)
        prior_prob = p / sum(p)
        labels = np.array([np.argmax(prior_prob[:, i]) for i in range(len(data))])  ## find max of prob
        data_cluster = [data[labels == i] for i in range(n_components)]
        ln_likelihood = sum([np.log(sum(np.exp(ln_exp_pdf(data[i], args=[f, m]).ravel()))) for i in range(len(data))])

        self.ln_likelihood = ln_likelihood
        return labels, data_cluster, ln_likelihood

    def plot_EM_results(self, save=False, path='output.png'):
        f = self.f
        m = self.m
        s = self.s
        fig, axs = plt.subplots(3, sharex=True)
        axs[-1].set_xlabel('iteration', fontsize=15)
        self.__plot_EM_result(m, axs[0], ylabel='mean')
        self.__plot_EM_result(s, axs[1], ylabel='std')
        self.__plot_EM_result(f, axs[2], ylabel='fraction')
        if save == True:
            save_img(fig, path)
        return fig

    ##  plot the survival function
    def plot_fit_exp(self, xlim=None, ylim=[0,1], save=False, path='output.png'):
        data = self.data
        f = self.f[-1,:]
        m = self.m[-1,:]
        s = self.s[-1,:]
        n_components = self.n_components
        n_sample = len(data)

        data_series = pd.Series(data.ravel())
        E = pd.Series(np.ones(len(data))) ## 1 = death
        kmf = KaplanMeierFitter()
        kmf.fit(data_series, event_observed=E)
        fig, ax = plt.subplots()
        kmf.plot_survival_function()
        ax.get_legend().remove() ## remove legend

        data_std_new = np.std(data)
        x = np.arange(0.01, max(data) + 3*data_std_new, 0.01)
        y_fit = exp_survival(x, args=[f, m])
        for i in range(n_components):
            ax.plot(x, y_fit[i, :], '-')
        ax.plot(x, sum(y_fit), 'r--')
        ax.set_xlabel('dwell time (s)', fontsize=15)
        ax.set_ylabel('probability density (1/$\mathregular{s^2}$)', fontsize=15)
        # xlim = [0, np.mean(data)+2*data_std_new]
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if save == True:
            save_img(fig, path)
        return fig

    ##  plot data histogram and its gaussian EM (GMM) results
    def plot_fit_gauss(self, save=False, path='output.png', scatter=False):
        data = self.data
        data_cluster = self.data_cluster
        f = self.f[-1,:]
        m = self.m[-1,:]
        s = self.s[-1,:]
        n_components = self.n_components
        data_std_new = np.std(data)
        x = np.arange(0, max(data) + data_std_new, 0.005)
        y_fit = oneD_gaussian(x, args=[f, m, s])

        if scatter==False:
            bin_number = np.log2(len(data)).astype('int') + 3
            pd, center, fig, ax = binning(data, bin_number)  # plot histogram
            for i in range(n_components):
                ax.plot(x, y_fit[i, :], '-', color=self.__colors_order()[i])
            ax.plot(x, sum(y_fit), 'r-')
            ax.set_xlabel('step size (nm)', fontsize=15)
            ax.set_ylabel('probability density (1/$\mathregular{nm^2}$)', fontsize=15)
        else:
            fig, ax = plt.subplots()
            for i in range(n_components):
                ax.plot(x, y_fit[i, :], '-', color=self.__colors_order()[i])
            ax.plot(x, sum(y_fit), 'r--')
            ax.set_xlabel('step size (nm)', fontsize=15)
            ax.set_ylabel('probability density (1/$\mathregular{nm^2}$)', fontsize=15)
            for i,x in enumerate(data_cluster):
                ax.plot(x, np.zeros(len(x)), 'o', markersize=4, color=self.__colors_order()[i])

        if save == True:
            save_img(fig, path)
        return fig

    def __AIC(self):
        ln_likelihood = self.ln_likelihood
        n_components = self.n_components
        AIC = -2 * ln_likelihood + (n_components * 3 - 1) * 2
        return AIC

    def __BIC(self):
        n_samples = len(self.data)
        ln_likelihood = self.ln_likelihood
        n_components = self.n_components
        BIC = -2 * ln_likelihood + (n_components * 3 - 1) * np.log(n_samples)
        return BIC

    ##  initialize mean, std and fraction for GMM
    def __init_GMM(self, n_components):
        self.n_components = n_components
        data = self.data.reshape((-1, 1))
        f, m, s = self.__get_f_m_s_kmeans(data)
        loop = 0
        improvement = 10
        return f, m, s, loop, improvement

    ##  initialize parameters for Poisson EM
    def __init_PEM(self, n_components):
        data = self.data
        self.n_components = n_components
        mean = np.mean(data)
        std = np.std(data)
        f = np.ones(n_components) / n_components
        tau = np.linspace(abs(mean - 0.5 * std), mean + 0.5 * std, n_components)
        s = tau.copy()
        loop = 0
        improvement = 10
        return f, tau, s, loop, improvement

    def __get_f_m_s_kmeans(self, data):
        n_sample = len(data)
        n_components = self.n_components
        labels = KMeans(n_clusters=n_components).fit(data).labels_
        data_cluster = [data[labels == i] for i in range(n_components)]
        m = np.array([np.mean(data) for data in data_cluster])
        index = np.argsort(m)
        f = np.array([len(data) / n_sample for data in data_cluster])[index]
        m = m[index]
        s = np.array([np.std(data) for data in data_cluster])[index]
        self.f_i = f
        self.m_i = m
        self.s_i = s
        return f, m, s

    ##  calculate the probability belonging to each cluster, (m,s)
    def __weighting(self, *args, function):
        """Calculate prior probability of each data point
        Parameters
        ----------
        function : use log function
        f, m, s : growing array, (n,)
            fractions, mean, std
        n_components : int
            Number of components.
        Returns
        -------
        prior_prob : array, (n_components, n_samples)

        """
        data = self.data
        n_components = self.n_components
        para = []
        for arg in args:
            para += [arg[-n_components:]]
        p = np.exp(function(data, args=para)) ##(n_components, n_samples)
        prior_prob = p / sum(p)
        self.p = p
        self.prior_prob = prior_prob
        return prior_prob


    ##  update mean, std and fraction using matrix multiplication, (n_feture, n_sample) * (n_sample, 1) = (n_feture, 1)
    def __update_f_m_s(self, prior_prob, f, m, s):
        """M-step
        Parameters
        ----------
        f, m, s : growing array, (n,)
            fractions, mean, std
        n_components : int
            Number of components.
        data(n_sample, 1) - m(n_components,) : array, (n_sample, n_components)
        Returns
        -------
        prior_prob : array, (n_components, n_samples)

        """

        data = self.data
        n_sample = len(data)
        # n_components = self.n_components

        f_new = np.sum(prior_prob, axis=1) / n_sample
        m_new = np.matmul(prior_prob, data).ravel() / np.sum(prior_prob, axis=1)
        s_new = np.sqrt( np.matmul(prior_prob, data ** 2).ravel()/(np.sum(prior_prob, axis=1)) - m_new**2 )

        f, m, s = self.__append_arrays([f,f_new], [m,m_new], [s,s_new])
        self.f = f
        self.m = m
        self.s = s
        return f, m, s

    def __append_arrays(self, *args):
        arrays = []
        for arg in args:
            arrays += [np.append(arg[0], arg[1])]
        return arrays

    ##  calculate max improvement among all parameters
    def __cal_improvement(self, *args):
        ##  arg: [f, m, s], not reshaped array
        """Calculate prior probability of each data point
                Parameters
                ----------
                f, m, s : growing array, (n,)
                    fractions, mean, std
                Returns
                -------
                prior_prob : array, (n_components, n_samples)

        """
        n_components = self.n_components
        improvement = []
        for arg in args:
            # arg = np.array(arg)
            arg_old = arg[-2*n_components:-n_components] ##  last 2n to last n
            arg_new = arg[-n_components:]
            diff = abs(arg_new - arg_old)
            improvement = max(np.append(improvement, diff)) ## take max of all args diff
        return improvement

    ##  reshape all arrays to (loop,n_components)
    def __reshape_all(self, *args, n_rows, n_cols):
        results = []
        for arg in args:
            results += [np.reshape(arg, (n_rows, n_cols))]
        return results

    def __plot_EM_result(self, result, ax, xlabel='iteration', ylabel='value'):
        n_feature = result.shape[1]
        iteration = result.shape[0]
        for i in range(n_feature):
            ax.plot(np.arange(0, iteration), result[:, i], '-o', color=self.__colors_order()[i])
        ax.set_ylabel(f'{ylabel}', fontsize=15)

    def __colors_order(self):
        colors = ['yellowgreen', 'seagreen', 'dodgerblue', 'darkslateblue', 'indigo', 'black']
        colors = ['green', 'royalblue', 'sienna', 'gray', 'black']
        return colors


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
        if s <= 1:
            s = 1
        y += [f*1/s/np.sqrt(2*math.pi)*np.exp(-(x-xm)**2/2/s**2)]
    y = np.array(y)
    return y.reshape(y.shape[0], y.shape[1])

def ln_oneD_gaussian(x, args):
    f = np.array(args[0])
    xm = np.array(args[1])
    s = np.array(args[2])
    lny = []
    for f, xm, s in zip(f, xm, s):
        if s <= 1:
            s = 1
        lny += [np.log(f) - np.log(s) - 1/2*np.log(2*math.pi) - (x-xm)**2/2/s**2]
    lny = np.array(lny)
    return lny.reshape(lny.shape[0], lny.shape[1])

def exp_survival(t, args):
    f = np.array(args[0])
    tau = np.array(args[1])
    y = []
    for f,tau in zip(f,tau):
        y += [f * np.exp(-t / tau)]
    y = np.array(y)
    return y.reshape(y.shape[0], y.shape[1])


##  args: list
def exp_pdf(t, args):
    f = np.array(args[0])
    tau = np.array(args[1])
    y = []
    for f,tau in zip(f,tau):
        y += [f * 1 / tau * np.exp(-t / tau)]
    y = np.array(y)
    return y.reshape(y.shape[0], y.shape[1])

def ln_exp_pdf(t, args):
    f = np.array(args[0])
    tau = np.array(args[1])
    lny = []
    for f,tau in zip(f,tau):
        lny += [np.log(f/tau) - t/tau]
    lny = np.array(lny)
    return lny.reshape(lny.shape[0], lny.shape[1])