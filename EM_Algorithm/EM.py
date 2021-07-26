
from matplotlib import rcParams
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial"]
rcParams.update({'font.size': 12})
# import matplotlib
# matplotlib.use('Agg')
from basic.binning import binning, scatter_hist
from basic.math_fn import to_1darray, oneD_gaussian, ln_oneD_gaussian, exp_survival, ln_exp_pdf, ln_gau_exp_pdf, exp_gauss_2d

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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
        self.s_lower = 1

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
        self.para_progress = [f, m, s]
        self.para_final = [f[-1], m[-1], s[-1]]
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
        data = self.data
        self.n_components = n_components
        self.tolerance = tolerance
        ##  initialize EM parameters
        f, m, s, loop, improvement = self.__init_GMM(data, n_components=n_components, rand_init=rand_init)
        converged = improvement < tolerance
        while (loop < 20 or ~converged) and loop < 500:
            prior_prob = self.__weighting(f, m, s, function=ln_oneD_gaussian)
            f, m, s = self.__update_f_m_s(data, prior_prob, f, m, s)
            improvement = self.__cal_improvement(f, m, s)
            converged = improvement < tolerance
            loop += 1
        f, m, s = self.__reshape_all(f, m, s, n_rows=loop+1, n_cols=n_components)
        self.para_progress = [f, m, s]
        m_f, f_f, s_f = self.__sort_according(m[-1], f[-1], s[-1])
        self.para_final = [f_f, m_f, s_f]
        para = self.para_final
        self.__cal_LLE(data, function=ln_oneD_gaussian, para=para)
        converged = np.array([converged] * n_components)
        self.converged = converged
        # labels, data_cluster = self.predict(data, ln_oneD_gaussian, paras=[f.ravel(), m.ravel(), s.ravel()])
        return f_f, m_f, s_f, converged

    def PEM(self, n_components, tolerance=1e-2, rand_init=False):
        data = self.data
        self.n_components = n_components
        self.tolerance = tolerance
        f, tau, s, loop, improvement = self.__init_PEM(data, n_components=n_components, rand_init=rand_init)
        converged = improvement < tolerance
        while (loop < 20 or ~converged) and loop < 500:
            prior_prob = self.__weighting(f, tau, function=ln_exp_pdf)
            f, tau, s = self.__update_f_m_s(data, prior_prob, f, tau, s)
            improvement = self.__cal_improvement(f, tau)
            converged = improvement < tolerance
            loop += 1
        f, tau, s = self.__reshape_all(f, tau, s, n_rows=loop+1, n_cols=n_components)
        self.para_progress = [f, tau, s]
        tau_f, f_f, s_f = self.__sort_according(tau[-1], f[-1], s[-1])
        self.para_final = [f_f, tau_f]
        para = self.para_final
        ln_likelihood = self.__cal_LLE(data, function=ln_exp_pdf, para=para)
        converged = np.array([converged] * n_components)
        self.converged = converged
        # labels, data_cluster = self.predict(data, ln_exp_pdf, paras=[f.ravel(), tau.ravel()])
        return f_f, tau_f, s_f, converged, ln_likelihood

    def GPEM(self, n_components, tolerance=1e-2, rand_init=False):
        data = self.data ## (n_samples, 2)
        x = data[:, 0] ## Gaussian R.V.
        y = data[:, 1] ## Poisson R.V.
        self.tolerance = tolerance
        ##  initialize EM parameters
        f1, m, s1, loop, improvement = self.__init_GMM(data[:,0], n_components=n_components, rand_init=rand_init)
        f2, tau, s2, loop, improvement = self.__init_PEM(data[:,1], n_components=n_components, rand_init=rand_init)
        converged = improvement < tolerance
        while (loop < 20 or ~converged) and loop < 500:
            prior_prob = self.__weighting(f1, m, s1, tau, function=ln_gau_exp_pdf)
            f1, m, s1 = self.__update_f_m_s(data[:,0].reshape(-1,1), prior_prob, f1, m, s1)
            f2, tau, s2 = self.__update_f_m_s(data[:,1].reshape(-1,1), prior_prob, f2, tau, s2)
            improvement = self.__cal_improvement(f1, m, s1, tau)
            converged = improvement < tolerance
            loop += 1
        f1, m, s1, tau = self.__reshape_all(f1, m, s1, tau, n_rows=loop+1, n_cols=n_components)
        self.para_progress = [f1, m, s1, tau]
        m_f, f_f, s_f, tau_f = self.__sort_according(m[-1], f1[-1], s1[-1], tau[-1])
        self.para_final = [f_f, m_f, s_f, tau_f]
        para = self.para_final
        ln_likelihood = self.__cal_LLE(data, function=ln_gau_exp_pdf, para=para)
        converged = np.array([converged] * n_components)
        self.converged = converged
        # labels, data_cluster = self.predict(data, function=ln_gau_exp_pdf, paras=para)
        return f_f, m_f, s_f, tau_f, converged, ln_likelihood

    ## set given m
    def GPEM_set(self, n_components, m_set, tolerance=1e-2, rand_init=False):
        data = self.data ## (n_samples, 2)
        x = data[:, 0] ## Gaussian R.V.
        y = data[:, 1] ## Poisson R.V.
        self.tolerance = tolerance
        ##  initialize EM parameters
        m_fix = m_set.copy()
        f1, m, s1, loop, improvement = self.__init_GMM(data[:,0], n_components=n_components, rand_init=rand_init)
        f2, tau, s2, loop, improvement = self.__init_PEM(data[:,1], n_components=n_components, rand_init=rand_init)
        converged = improvement < tolerance
        while (loop < 20 or ~converged) and loop < 500:
            prior_prob = self.__weighting(f1, m_fix, s1, tau, function=ln_gau_exp_pdf)
            m_fix = np.append(m_fix, m_set)
            f1, m1_notuse, s1 = self.__update_f_m_s(data[:,0].reshape(-1,1), prior_prob, f1, m_fix, s1)
            f2, tau, s2 = self.__update_f_m_s(data[:,1].reshape(-1,1), prior_prob, f2, tau, s2)
            improvement = self.__cal_improvement(f1, m_fix, s1, tau)
            converged = improvement < tolerance
            loop += 1
        f1, m_fix, s1, tau = self.__reshape_all(f1, m_fix, s1, tau, n_rows=loop+1, n_cols=n_components)
        self.para_progress = [f1, m_fix, s1, tau]
        m_fix_f, f_f, s_f, tau_f = self.__sort_according(m_fix[-1], f1[-1], s1[-1], tau[-1])
        self.para_final = [f_f, m_fix_f, s_f, tau_f]
        para = self.para_final
        ln_likelihood = self.__cal_LLE(data, function=ln_gau_exp_pdf, para=para)
        converged = np.array([converged] * n_components)
        self.converged = converged
        # labels, data_cluster = self.predict(data, function=ln_gau_exp_pdf, paras=para)
        return f_f, m_fix_f, s_f, tau_f, converged, ln_likelihood

    ##  iteratively find lowest BIC or AIC value
    def opt_components_iter(self, iteration=10, tolerance=1e-2, mode='GMM', criteria='BIC', figure=False, figsize=(10, 10)):
        n_all, c_all = [], []
        for i in range(iteration):
            n = self.opt_components(tolerance=tolerance, mode=mode, criteria=criteria, figure=figure, figsize=figsize)
            if criteria == 'AIC':
                c = self.AIC_owns[n - 1]
            else:
                c = self.BIC_owns[n - 1]
            n_all = np.append(n_all, n)
            c_all = np.append(c_all, c)
        index = np.argmin(c_all)
        return int(n_all[index])


    def opt_components(self, tolerance=1e-2, mode='GMM', criteria='BIC', figure=False, figsize=(10,10)):
        self.mode = mode
        ##  find best n_conponents
        data = self.data
        BICs, AICs = [], []
        BIC_owns, AIC_owns = [], []
        LLE = []
        n_clusters = np.arange(1, 6)
        for c in n_clusters:
            if mode == 'GMM':
                self.GMM(n_components=c, tolerance=tolerance, rand_init=True)
                gmm = GaussianMixture(n_components=c, tol=tolerance).fit(data)
                BICs += [gmm.bic(data)]
                AICs += [gmm.aic(data)]
            elif mode == 'PEM':
                self.PEM(n_components=c, tolerance=tolerance, rand_init=True)
            else:
                self.GPEM(n_components=c, tolerance=tolerance, rand_init=True)

            BIC_owns += [self.__BIC()]
            AIC_owns += [self.__AIC()]
            LLE += [self.ln_likelihood]

        BIC_owns, AIC_owns = to_1darray(BIC_owns, AIC_owns)
        ##  get optimal components
        if criteria=='AIC':
            opt_components = n_clusters[np.argmin(AIC_owns[~np.isnan(AIC_owns)])]
            if figure == True:
                fig, ax = plt.subplots(figsize=figsize)
                ax.plot(n_clusters, AIC_owns, '--o')
                ax.set_xlabel('n_components')
                ax.set_ylabel('AIC')
        else:
            opt_components = n_clusters[np.argmin(BIC_owns[~np.isnan(BIC_owns)])]
            if figure == True:
                fig, ax = plt.subplots(figsize=figsize)
                ax.plot(n_clusters, BIC_owns, '--o')
                ax.set_xlabel('n_components')
                ax.set_ylabel('BIC')
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
        para_progress = self.para_progress
        n_para = len(para_progress)
        fig, axs = plt.subplots(n_para, sharex=True, figsize=(10,6))
        axs[-1].set_xlabel('iteration', fontsize=22)
        names = ['fraction', 'mean', 'std', 'tau']
        for i in range(n_para):
            self.__plot_EM_result(para_progress[i], axs[i], ylabel=names[i])
        if save == True:
            save_img(fig, path)
        return fig

    ##  plot Gaussian-Poisson contour plot
    def plot_gp_contour(self, xlim=None, ylim=None, save=False, path='output.png', xlabel='Step-size (count)', ylabel='Dwell-time (s)'):
        data = self.data
        paras = self.para_final
        labels, data_cluster = self.predict(data, function=ln_gau_exp_pdf, paras=paras)

        x = np.linspace(min(data[:,0]), max(data[:,0]), 100)
        t = np.linspace(min(data[:,1]), max(data[:,1]), 100)
        x_mesh, t_mesh = np.meshgrid(x, t)
        x_t = np.array([x_mesh.ravel(), t_mesh.ravel()]).T
        data_fitted = ln_gau_exp_pdf(x_t, args=paras)
        fig, ax = plt.subplots(figsize=(10,8))
        # cmaps = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges']
        cmaps = ['Greens', 'Blues', 'Reds', 'Purples', 'summer',  'copper']

        for i,fit in enumerate(data_fitted):
            c1 = self.__colors_order()[i]
            ax.plot(data_cluster[i][:, 0], data_cluster[i][:, 1], 'o', color=c1, markersize=3)
        for i,fit in enumerate(data_fitted):
            ax.contour(x_mesh, t_mesh, np.exp(fit).reshape(len(x), len(t)), levels=5, cmap=cmaps[i], linewidths=3)
        ax.set_xlabel(xlabel, fontsize=22)
        ax.set_ylabel(ylabel, fontsize=22)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        plt.tight_layout()
        plt.show()
        if save == True:
            save_img(fig, path)


    def plot_gp_contour_2hist(self, xlim=None, ylim=None, figsize=(10,10), fontsize=12, bins_x=10, bins_y=10,
                              save=False, path='2d_scatter.png'):
        data = self.data
        paras = self.para_final
        labels, data_cluster = self.predict(data, function=ln_gau_exp_pdf, paras=paras)

        x = np.linspace(min(data[:,0]), max(data[:,0]+10), 100)
        t = np.linspace(min(data[:,1]), max(data[:,1]+10), 100)
        x_mesh, t_mesh = np.meshgrid(x, t)
        x_t = np.array([x_mesh.ravel(), t_mesh.ravel()]).T
        data_fitted = ln_gau_exp_pdf(x_t, args=paras)
        # start with a square Figure
        fig = plt.figure(figsize=figsize)

        # Add a gridspec with two rows and two columns and a ratio of 2 to 7 between
        # the size of the marginal axes and the main axes in both directions.
        # Also adjust the subplot parameters for a square plot.
        gs = fig.add_gridspec(2, 2, width_ratios=(7, 2), height_ratios=(2, 7),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.05, hspace=0.05)

        ax = fig.add_subplot(gs[1, 0])
        # the scatter plot:
        colors = ['green', 'royalblue', 'sienna', 'magenta', 'darkgreen', 'darkslateblue', 'maroon', 'black']
        cmaps = ['Greens', 'Blues', 'Reds', 'Purples', 'summer',  'copper']
        for i,fit in enumerate(data_fitted):
            c1 = self.__colors_order()[i]
            ax.plot(data_cluster[i][:, 0], data_cluster[i][:, 1], 'o', color=c1, markersize=3)
        for i,fit in enumerate(data_fitted):
            ax.contour(x_mesh, t_mesh, np.exp(fit).reshape(len(x), len(t)), levels=5, cmap=cmaps[i], linewidths=3)
        ax.set_xlabel('Step-size (count)', fontsize=fontsize)
        ax.set_ylabel('Dwell-time (s)', fontsize=fontsize)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.spines[:].set_linewidth('1.5')  ## xy, axis width
        ax.tick_params(width=1.5)  ## tick width

        ## gaussian plot
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_histx.hist(data[:,0], bins=bins_x, color='grey', edgecolor="white", density=True)
        y_fit = oneD_gaussian(x, args=paras[:3])
        for i in range(len(paras[0])):
            ax_histx.plot(x, y_fit[i, :], '-', color=self.__colors_order()[i])
        ax_histx.plot(x, sum(y_fit), 'r-')
        ax_histx.spines[:].set_linewidth('1.5')  ## xy, axis width
        ax_histx.tick_params(width=1.5)  ## tick width
        ax_histx.set_yticks([])
        ## survival plot
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
        data_series = pd.Series(data[:,1].ravel())
        E = pd.Series(np.ones(len(data[:,1]))) ## 1 = death
        kmf = KaplanMeierFitter()
        kmf.fit(data_series, event_observed=E)
        kmf.plot_survival_function(ax=ax_histy)

        y_fit = exp_survival(x, args=[paras[0]]+[paras[3]])
        for i in range(len(paras[0])):
            ax_histy.plot(y_fit[i, :], x, '-', color=self.__colors_order()[i])
        ax_histy.plot(sum(y_fit), x, 'r--')
        ax_histy.spines[:].set_linewidth('1.5')  ## xy, axis width
        ax_histy.tick_params(width=1.5)  ## tick width
        ax_histy.set_xticks([])

        ax_histy.get_legend().remove() ## remove legend

        # ax_histy.hist(data[:,1], bins=bins_y, orientation='horizontal', color='grey', edgecolor="white")
        plt.show
        if save == True:
            save_img(fig, path)


    def plot_gp_surface(self, x_end=20, t_end=10, xlabel='Step-size (count)', ylabel='Dwell-time (s)', zlabel='probability density'):
        data = self.data
        paras = self.para_final

        x = np.linspace(0, x_end, 100)
        t = np.linspace(0, t_end, 100)
        X, Y = np.meshgrid(x, t)
        Z = exp_gauss_2d(X, Y, *paras)

        fig = plt.figure(figsize=(10,8))
        ax = plt.axes(projection="3d")
        ax.plot_wireframe(X, Y, Z, color='green')

        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, Z,
                        edgecolor='none', alpha=0.3)
        cset = ax.contour(X, Y, Z, zdir='z', offset=-0.5)
        cset = ax.contour(X, Y, Z, zdir='x', offset=0, levels=1)
        cset = ax.contour(X, Y, Z, zdir='y', offset=t_end, levels=1)

        Z_data = exp_gauss_2d(data[:,0], data[:,1], *paras)
        ax.scatter(data[:,0], data[:,1], -0.5*np.ones(data.shape[0]), c=Z_data, linewidth=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_xlim(0, x_end)
        ax.set_ylim(0, t_end)
        ax.set_zlim(-0.5, 1.5)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.zaxis.set_major_locator(MaxNLocator(4))


    ##  plot the survival function
    def plot_fit_exp(self, xlim=None, ylim=[0,1], save=False, path='output.png',
                     xlabel='Dwell-time (s)', ylabel='Survival', figsize=(10,10),
                     para=None, fontsize=12, remove_xtick=False, remove_ytick=False, line_width=1.5):
        data = self.data
        if para == None:
            para = self.para_final
        n_components = self.n_components
        fig, ax = self.__plot_survival(data, figsize)
        x = np.arange(0.01, max(data) + 3*np.std(data), 0.01)
        y_fit = exp_survival(x, args=para)
        for i in range(len(para[0])):
            ax.plot(x, y_fit[i, :], '-', color=self.__colors_order()[i])
        ax.plot(x, sum(y_fit), 'r--')
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.spines[:].set_linewidth(f'{line_width}')  ## xy, axis width
        ax.tick_params(width=line_width)  ## tick width
        if remove_xtick:
            ax.set_xticks([])
        if remove_ytick:
            ax.set_yticks([])
        plt.show()
        if save == True:
            save_img(fig, path)
        return fig, ax

    ##  plot data histogram and its gaussian EM (GMM) results
    def plot_fit_gauss(self, xlim=None, ylim=None, save=False, path='output.png', scatter=False,
                       figsize=(10,8), color="grey", fontsize=22, xlabel='Step-size (count)',
                       ylabel='Probability density (1/$\mathregular{count}$)', para=None):
        data = self.data
        if para == None:
            para = self.para_final
        labels, data_cluster = self.predict(data, ln_oneD_gaussian, paras=para)
        n_components = self.n_components
        x = np.arange(0, max(data) + np.std(data), 0.001)
        y_fit = oneD_gaussian(x, args=para)

        if scatter==False:
            bin_number = np.log2(len(data)).astype('int')
            pd, center, fig, ax = binning(data, bin_number, figsize=figsize, color=color, fontsize=fontsize)  # plot histogram
            for i in range(len(para[0])):
                ax.plot(x, y_fit[i, :], '-', color=self.__colors_order()[i])
            ax.plot(x, sum(y_fit), 'r-')

        else:
            bin_number = np.log2(len(data)).astype('int')
            pd, center, fig, ax = binning(data, bin_number)  # plot histogram
            for i in range(n_components):
                ax.plot(x, y_fit[i, :], '-', color=self.__colors_order()[i])
            ax.plot(x, sum(y_fit), 'r--')

            for i,x in enumerate(data_cluster):
                ax.plot(x, np.zeros(len(x)), 'o', markersize=5, color=self.__colors_order()[i])
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.spines[:].set_linewidth('1.5')  ## xy, axis width
        ax.tick_params(width=1.5)  ## tick width
        plt.show()
        if save == True:
            save_img(fig, path)
        return fig

    ##  plot Kaplan_Meier method
    def __plot_survival(self, data, figsize=(10,8)):
        data_series = pd.Series(data.ravel())
        E = pd.Series(np.ones(len(data))) ## 1 = death
        kmf = KaplanMeierFitter()
        kmf.fit(data_series, event_observed=E)
        fig, ax = plt.subplots(figsize=figsize)
        kmf.plot_survival_function()
        ax.get_legend().remove() ## remove legend
        plt.show()
        self.kmf = kmf
        return fig, ax

    ##  calculate log-likelihood of given parameters, function is log-function
    def __cal_LLE(self, data, function, para):
        ln_likelihood = sum([np.log(sum(np.exp(function(data[i,:].reshape(1,-1), args=para).ravel()))) for i in range(data.shape[0])])
        ln_likelihood = np.array(ln_likelihood, ndmin=1)
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
                s[i] = random.random()*np.std(data) + 0.5
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
            f = np.ones(n_components)
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
        s_lower = self.s_lower
        n_sample = len(data)
        # n_components = self.n_components
        f_new = np.sum(prior_prob, axis=1) / n_sample
        m_new = np.matmul(prior_prob, data).ravel() / np.sum(prior_prob, axis=1)
        s_new = np.sqrt( np.matmul(prior_prob, data**2).ravel()/(np.sum(prior_prob, axis=1)) - m_new**2 )
        if any(s_new <= s_lower) or any(np.isnan(s_new)):
            s_new[s_new <= s_lower] = random.random()+0.5
            s_new[np.isnan(s_new)] = random.random()+0.5

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
        ax.set_ylabel(f'{ylabel}', fontsize=22)
        plt.show()

    def __colors_order(self):
        colors = ['yellowgreen', 'seagreen', 'dodgerblue', 'darkslateblue', 'indigo', 'black']
        colors = ['green', 'royalblue', 'sienna', 'magenta', 'darkgreen', 'darkslateblue', 'maroon', 'black']
        return colors

