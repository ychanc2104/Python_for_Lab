
from matplotlib import rcParams
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial"]
rcParams.update({'font.size': 18})
from basic.binning import binning
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from basic.file_io import save_img
from lifelines import KaplanMeierFitter
import pandas as pd
import random


### data: (n,1)-array
class EM:
    def __init__(self, data, dim=1):
        self.data = data.reshape(-1, dim)

    def skGMM(self, n_components, tolerance=10e-5):
        self.n_components = n_components
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


    def GMM(self, n_components, tolerance=1e-2, rand_init=False):
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
        self.n_components = n_components
        data = self.data
        self.tolerance = tolerance
        ##  initialize EM parameters
        f, m, s, loop, improvement = self.__init_GMM(data, n_components=n_components, rand_init=rand_init)
        f1 = np.ones(len(m))
        while (loop < 10 or improvement > tolerance) and loop < 500:
            prior_prob = self.__weighting(f1, m, s, function=ln_oneD_gaussian)
            f, m, s = self.__update_f_m_s(data, prior_prob, f, m, s)
            improvement = self.__cal_improvement(f, m, s)
            loop += 1
            f1 = np.ones(len(m))
        self.__weighting(f1, m, s, function=ln_oneD_gaussian) # update prior prob
        f, m, s = self.__reshape_all(f, m, s, n_rows=loop+1, n_cols=n_components)
        self.f = f
        self.m = m
        self.s = s
        ln_likelihood = self.__cal_LLE(data, function=ln_oneD_gaussian, para=[f[-1].ravel(), m[-1].ravel(), s[-1].ravel()])
        labels, data_cluster = self.predict(data, ln_oneD_gaussian, paras=[f.ravel(), m.ravel(), s.ravel()])
        return f, m, s, labels, data_cluster

    def PEM(self, n_components, tolerance=1e-2, rand_init=False):
        self.n_components = n_components
        data = self.data
        self.tolerance = tolerance

        f, tau, s, loop, improvement = self.__init_PEM(data, n_components=n_components, rand_init=rand_init)
        f1 = np.ones(len(tau))
        while (loop < 10 or improvement > tolerance) and loop < 500:
            prior_prob = self.__weighting(f1, tau, function=ln_exp_pdf)
            f, tau, s = self.__update_f_m_s(data, prior_prob, f, tau, s)
            improvement = self.__cal_improvement(f, tau)
            loop += 1
            f1 = np.ones(len(tau))
        self.__weighting(f1, tau, function=ln_exp_pdf)
        f, tau, s = self.__reshape_all(f, tau, s, n_rows=loop+1, n_cols=n_components)
        self.f = f
        self.m = tau
        self.s = s
        ln_likelihood = self.__cal_LLE(data, function=ln_exp_pdf, para=[f[-1].ravel(), tau[-1].ravel()])

        labels, data_cluster = self.predict(data, ln_exp_pdf, paras=[f.ravel(), tau.ravel()])
        return f, tau, s, labels, data_cluster

    def GPEM(self, n_components, tolerance=1e-2, rand_init=False):
        data = self.data ## (n_samples, 2)
        x = data[:, 0] ## Gaussian R.V.
        y = data[:, 1] ## Poisson R.V.
        self.tolerance = tolerance
        ##  initialize EM parameters
        f1, m, s1, loop, improvement = self.__init_GMM(data[:,0], n_components=n_components, rand_init=rand_init)
        f2, tau, s2, loop, improvement = self.__init_PEM(data[:,1], n_components=n_components, rand_init=rand_init)
        f11 = np.ones(len(m))
        f22 = np.ones(len(tau))
        while (loop < 10 or improvement > tolerance) and loop < 500:
            prior_prob = self.__weighting(f11, m, s1, f22, tau, function=ln_gau_exp_pdf)
            f1, m, s1 = self.__update_f_m_s(data[:,0].reshape(-1,1), prior_prob, f1, m, s1)
            f2, tau, s2 = self.__update_f_m_s(data[:,1].reshape(-1,1), prior_prob, f2, tau, s2)
            improvement = self.__cal_improvement(m, s1, tau)
            loop += 1
            f11 = np.ones(len(m))
            f22 = np.ones(len(tau))
        self.__weighting(f11, m, s1, f22, tau, function=ln_gau_exp_pdf)
        f1, m, s1, f2, tau = self.__reshape_all(f1, m, s1, f2, tau, n_rows=loop+1, n_cols=n_components)
        self.f1 = f1
        self.m = m
        self.s1 = s1
        self.f2 = f2
        self.tau = tau
        self.s2 = s2
        para = [f1[-1].ravel(), m[-1].ravel(), s1[-1].ravel(), f2[-1].ravel(), tau[-1].ravel()]
        ln_likelihood = self.__cal_LLE(data, function=ln_gau_exp_pdf, para=para)

        # labels, data_cluster = self.predict(data, function=ln_gau_exp_pdf, paras=[f1,m,s1,f2,tau])
        return f1, m, s1, f2, tau


    def opt_components(self, tolerance=1e-2, mode='GMM', criteria='AIC', figure=False):
        self.mode = mode
        ##  find best n_conponents
        data = self.data
        BICs = []
        AICs = []
        BIC_owns = []
        AIC_owns = []
        LLE = []
        n_clusters = np.arange(1, 8)
        for c in n_clusters:
            if mode == 'GMM':
                f, tau, s, labels, data_cluster = self.GMM(n_components=c, tolerance=tolerance)
                gmm = GaussianMixture(n_components=c, tol=tolerance).fit(data)
                BICs += [gmm.bic(data)]
                AICs += [gmm.aic(data)]
            elif mode == 'PEM':
                f, tau, s, labels, data_cluster = self.PEM(n_components=c, tolerance=tolerance)
            else:
                f1, m, s1, f2, tau = self.GPEM(n_components=c, tolerance=tolerance)

            BIC_owns += [self.__BIC()]
            AIC_owns += [self.__AIC()]
            LLE += [self.ln_likelihood]
        if figure == True:
            plt.figure()
            plt.plot(n_clusters, BIC_owns, '--o')
            plt.title('BIC')
            plt.xlabel('n_components')
            plt.ylabel('BIC_owns')
            plt.figure()
            plt.plot(n_clusters, AIC_owns, '--o')
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
    def predict(self, data, function, paras):
        """predict data cluster
        Parameters
        ----------
        paras : list array, ex:[f,m,s] f,m,s : growing array, (n,)
        ln_likelihood : int
            Number of components.
        tolerance : float
            Convergence criteria
        data : array (n_samples, k)

        prior_prob: array (n_components, n_sample)

        Returns
        -------

        """

        n_components = self.n_components
        # paras = np.array(paras).reshape()
        paras = np.array(paras)[:, -n_components:] ## size to (n_paras, n_components)
        p = np.exp(function(data, args=paras)) ##(n_components, n_samples)
        prior_prob = p / sum(p)
        labels = np.array([np.argmax(prior_prob[:, i]) for i in range(len(data))])  ## find max of prob
        data_cluster = [data[labels == i] for i in range(n_components)]
        self.data_cluster = data_cluster
        # ln_likelihood = sum([np.log(sum(np.exp(function(data[i], args=paras).ravel()))) for i in range(len(data))])

        # self.ln_likelihood = ln_likelihood
        return labels, data_cluster

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

    ##  plot Gaussian-Poisson contour plot
    def plot_gp_contour(self, save=False, path='output.png'):
        data = self.data
        f1 = self.f1[-1,:].ravel()
        m = self.m[-1,:].ravel()
        s1 = self.s1[-1,:].ravel()
        f2 = self.f2[-1,:].ravel()
        tau = self.tau[-1,:].ravel()
        n_components = self.n_components
        paras = [f1, m, s1, f2, tau]
        labels, data_cluster = self.predict(data, function=ln_gau_exp_pdf, paras=paras)

        x = np.linspace(min(data[:,0]), max(data[:,0]), 100)
        t = np.linspace(min(data[:,1]), max(data[:,1]), 100)
        x_mesh, t_mesh = np.meshgrid(x, t)
        x_t = np.array([x_mesh.ravel(), t_mesh.ravel()]).T
        data_fitted = ln_gau_exp_pdf(x_t, args=paras)

        fig, ax = plt.subplots()
        # cmaps = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges']
        cmaps = ['Reds', 'cool', 'winter', 'Wistia', 'copper']

        for i,fit in enumerate(data_fitted):
            c1 = self.__colors_order()[i+2]
            ax.plot(data_cluster[i][:, 0], data_cluster[i][:, 1], 'o', color=c1, markersize=2)
        for i,fit in enumerate(data_fitted):
            c = self.__colors_order()[i]
            ax.contour(x_mesh, t_mesh, np.exp(fit).reshape(len(x), len(t)), levels=5, cmap=cmaps[i])

        ax.set_xlabel('step size', fontsize=22)
        ax.set_ylabel('dwell time', fontsize=22)
        plt.tight_layout()
        plt.show()




    ##  plot the survival function
    def plot_fit_exp(self, xlim=None, ylim=[0,1], save=False, path='output.png'):
        data = self.data
        f = self.f[-1,:]
        m = self.m[-1,:]
        s = self.s[-1,:]
        n_components = self.n_components
        fig, ax = self.__plot_survival(data)

        data_std_new = np.std(data)
        x = np.arange(0.01, max(data) + 3*data_std_new, 0.01)
        y_fit = exp_survival(x, args=[f, m])
        for i in range(n_components):
            ax.plot(x, y_fit[i, :], '-')
        ax.plot(x, sum(y_fit), 'r--')
        ax.set_xlabel('dwell time (s)', fontsize=22)
        ax.set_ylabel('survival', fontsize=22)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if save == True:
            save_img(fig, path)
        return fig

    ##  plot data histogram and its gaussian EM (GMM) results
    def plot_fit_gauss(self, xlim=None, ylim=None, save=False, path='output.png', scatter=False):
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
            ax.set_xlabel('step size (nm)', fontsize=22)
            ax.set_ylabel('probability density (1/$\mathregular{nm^2}$)', fontsize=22)
        else:
            fig, ax = plt.subplots()
            for i in range(n_components):
                ax.plot(x, y_fit[i, :], '-', color=self.__colors_order()[i])
            ax.plot(x, sum(y_fit), 'r--')
            ax.set_xlabel('step size (nm)', fontsize=22)
            ax.set_ylabel('probability density (1/$\mathregular{nm^2}$)', fontsize=22)
            for i,x in enumerate(data_cluster):
                ax.plot(x, np.zeros(len(x)), 'o', markersize=4, color=self.__colors_order()[i])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if save == True:
            save_img(fig, path)
        return fig

    ##  plot Kaplan_Meier method
    def __plot_survival(self, data):
        data_series = pd.Series(data.ravel())
        E = pd.Series(np.ones(len(data))) ## 1 = death
        kmf = KaplanMeierFitter()
        kmf.fit(data_series, event_observed=E)
        fig, ax = plt.subplots()
        kmf.plot_survival_function()
        ax.get_legend().remove() ## remove legend
        self.kmf = kmf
        return fig, ax

    ##  calculate log-likelihood of given parameters, function is log-function
    def __cal_LLE(self, data, function, para):
        ln_likelihood = sum([np.log(sum(np.exp(function(data[i,:].reshape(1,-1), args=para).ravel()))) for i in range(data.shape[0])])
        self.ln_likelihood = ln_likelihood
        return ln_likelihood


    def __AIC(self):
        mode = self.mode
        ln_likelihood = self.ln_likelihood
        n_components = self.n_components
        if mode == 'GMM':
            AIC = -2 * ln_likelihood + (n_components * 3 - 1) * 2
        elif mode == 'PEM':
            AIC = -2 * ln_likelihood + (n_components * 2 - 1) * 2
        else: ## GP-EM
            AIC = -2 * ln_likelihood + (n_components * 4 - 1) * 2

        return AIC

    def __BIC(self):
        mode = self.mode
        n_samples = len(self.data)
        ln_likelihood = self.ln_likelihood
        n_components = self.n_components
        if mode == 'GMM':
            BIC = -2 * ln_likelihood + (n_components * 3 - 1) * np.log(n_samples)
        elif mode == 'PEM':
            BIC = -2 * ln_likelihood + (n_components * 2 - 1) * np.log(n_samples)
        else:
            BIC = -2 * ln_likelihood + (n_components * 4 - 1) * np.log(n_samples)

        return BIC

    ##  initialize mean, std and fraction for GMM
    def __init_GMM(self, data, n_components, rand_init=False):

        self.n_components = n_components
        data = data.reshape(-1, 1)
        if rand_init==False:
            f, m, s = self.__get_f_m_s_kmeans(data)
        else:
            f = np.zeros(n_components)
            m = np.zeros(n_components)
            s = np.zeros(n_components)
            for i in range(n_components):
                f[i] = random.random()
                m[i] = random.random()*max(data)
                s[i] = random.random()*np.std(data)
            m, f, s = self.__sort_according(m, f, s) ## sort according to first array

        loop = 0
        improvement = 10
        return f, m, s, loop, improvement

    ##  initialize parameters for Poisson EM
    def __init_PEM(self, data, n_components, rand_init=False):
        # data = self.data
        data = data.reshape(-1, 1)
        self.n_components = n_components
        mean = np.mean(data)
        std = np.std(data)
        if rand_init==False:
            f = np.ones(n_components) / n_components
            tau = np.linspace(abs(mean - 0.5 * std), mean + 0.5 * std, n_components)
            s = tau.copy()
        else:
            f = np.zeros(n_components)
            tau = np.zeros(n_components)
            s = np.zeros(n_components)
            for i in range(n_components):
                f[i] = random.random()
                tau[i] = random.random()*max(data)
                s[i] = random.random()*np.std(data)
            tau, f, s = self.__sort_according(tau, f, s) ## sort according to first array

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
    def __update_f_m_s(self, data, prior_prob, f, m, s):
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
        # data = self.data
        n_sample = len(data)
        # n_components = self.n_components
        f_new = np.sum(prior_prob, axis=1) / n_sample
        m_new = np.matmul(prior_prob, data).ravel() / np.sum(prior_prob, axis=1)
        s_new = np.sqrt( np.matmul(prior_prob, data**2).ravel()/(np.sum(prior_prob, axis=1)) - m_new**2 )

        f, m, s = self.__append_arrays([f,f_new], [m,m_new], [s,s_new])
        self.f = f
        self.m = m
        self.s = s
        return f, m, s


    # def update_gau_poi(self, prior_prob, f1, m, s1, f2,):


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

    def __sort_according(self, *args):
        index = np.argsort(args[0])
        results = []
        for arg in args:
            arg = np.array(arg)
            results += [arg[index]]
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

def gau_exp_pdf(data, args):
    data = data.reshape(-1,2)
    x = data[:, 0]
    t = data[:, 1]
    f1 = np.array(args[0])
    xm = np.array(args[1])
    s1 = np.array(args[2])
    tau = np.array(args[4])
    y = []
    for f1, xm, s1, tau in zip(f1, xm, s1, tau):

        if s1 <= 0.5:
            s1 = 0.5
        if tau <= 0.01:
            tau = 0.01
        y += [f1*1/s1/np.sqrt(2 * math.pi)*np.exp(-(x - xm)**2/2/s1**2)*f1*1/tau*np.exp(-t/tau)]
    y = np.array(y)
    return y.reshape(y.shape[0], y.shape[1])


def ln_gau_exp_pdf(data, args):
    data = data.reshape(-1,2)
    x = data[:, 0]
    t = data[:, 1]
    f1 = np.array(args[0])
    xm = np.array(args[1])
    s1 = np.array(args[2])
    tau = np.array(args[4])
    lny = []
    for f1, xm, s1, tau in zip(f1, xm, s1, tau):

        if s1 <= 0.5:
            s1 = 0.5
        if tau <= 0.01:
            tau = 0.01
        lny += [np.log(f1)-np.log(s1)-1/2*np.log(2*math.pi)-(x-xm)**2/2/s1**2 + np.log(f1/tau) - t/tau]
    lny = np.array(lny)
    return lny.reshape(lny.shape[0], lny.shape[1])