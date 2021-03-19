
from EM_Algorithm.gen_poisson import gen_poisson
import numpy as np
from EM_Algorithm.EM import EM

if __name__ == "__main__":
    ##  produce data
    data = gen_poisson(tau=[0.1, 2], n_sample=[200, 200])
    # data =[0.2925 0.315  0.7875 1.0575 0.42   0.21   0.1125 0.105  0.1125 0.1725, 0.315  0.8175 0.18   0.1125 1.0275 0.2475 0.2475 0.51   2.28   0.72, 1.255  0.515  1.425  1.845  1.395  0.345  2.325  2.095  1.81   0.56, 1.065  0.875  0.65   2.525  0.665  1.38   2.17   1.125  0.245  1.38, 0.77   3.125  0.855  2.34   5.86   0.345  0.825  0.145  0.42   0.19, 0.375  0.345  0.735  1.76   1.115  0.46   1.285  0.605  3.43   0.765, 1.53   1.04   1.615  0.58   1.235  1.23   0.345  1.205  0.25   2.75, 0.595  0.835  1.385  4.225  0.6975 0.96   0.2475 0.18   0.3825 0.2325, 1.4025 0.1125 0.2625 0.42   0.3675 0.1425 0.5775 0.1575 0.105  0.0975]
    data = [0.2925,
            0.315,
            0.7875,
            1.0575,
            0.42,
            0.21,
            0.1125,
            0.105,
            0.1125,
            0.1725,
            0.315,
            0.8175,
            0.18,
            0.1125,
            1.0275,
            0.2475,
            0.2475,
            0.51,
            2.28,
            0.72,
            1.255,
            0.515,
            1.425,
            1.845,
            1.395,
            0.345,
            2.325,
            2.095,
            1.81,
            0.56,
            1.065,
            0.875,
            0.65,
            2.525,
            0.665,
            1.38,
            2.17,
            1.125,
            0.245,
            1.38,
            0.77,
            3.125,
            0.855,
            2.34,
            5.86,
            0.345,
            0.825,
            0.145,
            0.42,
            0.19,
            0.375,
            0.345,
            0.735,
            1.76,
            1.115,
            0.46,
            1.285,
            0.605,
            3.43,
            0.765,
            1.53,
            1.04,
            1.615,
            0.58,
            1.235,
            1.23,
            0.345,
            1.205,
            0.25,
            2.75,
            0.595,
            0.835,
            1.385,
            4.225,
            0.6975,
            0.96,
            0.2475,
            0.18,
            0.3825,
            0.2325,
            1.4025,
            0.1125,
            0.2625,
            0.42,
            0.3675,
            0.1425,
            0.5775,
            0.1575,
            0.105,
            0.0975,
            ]
    data = np.array(data)
    data = gen_poisson(tau=[0.5, 2, 10, 30], n_sample=[2000]*4)

    n_sample = len(data)
    tolerance = 1e-2
    ##  fit Poisson EM
    EM_p = EM(data)
    # n_components = 2
    n_components_p = EM_p.opt_components(tolerance=1e-2, mode='PEM', criteria='AIC', figure=True)
    f, tau, s, labels, data_cluster = EM_p.PEM(n_components_p, tolerance)
    EM_p.plot_EM_results()
    EM_p.plot_fit_exp(xlim=[0,50], ylim=[0,0.3])



    # ##  find best n_conponents
    # BICs = []
    # BIC_owns = []
    # AIC_owns = []
    # n_clusters = np.arange(1, 5)
    # for c in n_clusters:
    #     f_save, tau_save, BIC, AIC = exp_EM(data, n_components=c, tolerance=tolerance)
    #     BIC_owns += [BIC]
    #     AIC_owns += [AIC]
    #
    # plt.figure()
    # plt.plot(n_clusters, BIC_owns, 'o')
    # plt.title('BIC_owns')
    # plt.xlabel('n_components')
    # plt.ylabel('BIC_owns')
    #
    # plt.figure()
    # plt.plot(n_clusters, AIC_owns, 'o')
    # plt.title('AIC_owns')
    # plt.xlabel('n_components')
    # plt.ylabel('AIC_owns')
    #
    # ##  fit EM
    # n_components = n_clusters[np.argmin(AIC_owns)]
    # f_save, tau_save, BIC, AIC = exp_EM(data, n_components, tolerance=tolerance)
    #
    # ##  plot EM results(mean, std, fractio with iteration)
    # plot_EM_results_exp([f_save, tau_save], ['fraction', 'dwell time'])
    #
    # ##  plot data histogram and its gaussian EM (GMM) results
    # plot_fit_pdf(data, exp_pdf, [f_save[-1,:], tau_save[-1,:]])
