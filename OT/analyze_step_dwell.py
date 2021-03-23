
from matplotlib import rcParams
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial"]

from basic.select import get_files, select_folder
import scipy.io as sio
import numpy as np
from EM_Algorithm.EM import *
import matplotlib.pyplot as plt


if __name__ == '__main__':
        
    conc = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    # conc = [0.0, 1.0, 1.5, 2.0]
    # conc = [1.0]
    
    path_folder = select_folder()
    steps = []
    dwells = []
    tolerance = 1e-3

    all_fractions = []
    all_centers = []
    all_stds = []

    all_tau = []
    all_ftau = []
    all_stau = []

    all_2d_f = []
    all_2d_m = []
    all_2d_s = []
    all_2d_tau = []
    for c in conc:
        path_data = get_files(f'S5S1_{c}*.mat', dialog=False, path_folder=path_folder)
        step = []
        dwell = []
        for path in path_data:
            data = sio.loadmat(path)
            step = np.append(step, data['step'])
            dwell = np.append(dwell, [data['dwell']])

        ## get Gaussian EM results
        EM_g = EM(step)
        n_components_g = EM_g.opt_components(tolerance=1e-2, mode='GMM', criteria='BIC', figure=False)
        f, m, s, labels, data_cluster = EM_g.GMM(n_components_g)
        EM_g.plot_fit_gauss(scatter=True, xlim=[0,18])

        steps += [step]
        dwells += [dwell]

        all_fractions += [f[-1, :]]
        all_centers += [m[-1,:]]
        all_stds += [s[-1,:]]

    
        ## get poisson EM results
        EM_p = EM(dwell)
        n_components_p = EM_p.opt_components(tolerance=1e-2, mode='PEM', criteria='AIC', figure=False)
        f_tau, tau, s_tau, labels, data_cluster = EM_p.PEM(n_components_p)
        EM_p.plot_fit_exp(xlim=[0,10])
        all_ftau += [f_tau[-1, :]]
        all_tau += [tau[-1, :]]
        all_stau += [s_tau[-1,:]]

        ##  2D clustering
        step_dwell = np.array([step, dwell]).T
        # plt.plot(data_g, data_p, 'o')
        EM_gp = EM(step_dwell, dim=2)
        # n_components_gp = EM_gp.opt_components(tolerance=1e-2, mode='GP', criteria='BIC', figure=False)


        f1, m1, s1, f2, tau1 = EM_gp.GPEM(n_components=n_components_g, tolerance=1e-10, rand_init=True)
        para = [f1[-1].ravel(), m1[-1].ravel(), s1[-1].ravel(), f2[-1].ravel(), tau1[-1].ravel()]
        labels, data_cluster = EM_gp.predict(step_dwell, function=ln_gau_exp_pdf, paras=para)
        plt.figure()
        for x in data_cluster:
            plt.plot(x[:, 0], x[:, 1], 'o')
        all_2d_f += [f1[-1,:]]
        all_2d_m += [m1[-1,:]]
        all_2d_s += [s1[-1,:]]
        all_2d_tau += [tau1[-1,:]]


    
    
