
from matplotlib import rcParams
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial"]

from basic.select import get_files, select_folder
import scipy.io as sio
import numpy as np
from EM_Algorithm.EM import *
import matplotlib.pyplot as plt


if __name__ == '__main__':
    all_results = []
    for i in range(3):
        conc = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0] ## S5S1
        # conc = [1.0, 1.2, 1.5, 1.8, 2.0, 3.0, 4.0] ## m51 only
        # conc = [0.1, 0.2, 0.25, 0.5, 0.7, 0.8, 1.1, 1.2, 2.0]  ## EcRecA
        # conc = [0.0]
    
    
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
        results = []
        for c in conc:
            path_data = get_files(f'*{c}*.mat', dialog=False, path_folder=path_folder)
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
            EM_g.plot_fit_gauss(scatter=True, xlim=[0,18], save=True, path=f'{c}_gauss.png')
    
            steps += [step]
            dwells += [dwell]
    
            all_fractions += [f[-1, :]]
            all_centers += [m[-1,:]]
            all_stds += [s[-1,:]]
    
        
            ## get poisson EM results
            EM_p = EM(dwell)
            n_components_p = EM_p.opt_components(tolerance=1e-2, mode='PEM', criteria='BIC', figure=False)
            f_tau, tau, s_tau, labels, data_cluster = EM_p.PEM(n_components_p)
            EM_p.plot_fit_exp(xlim=[0,10], save=True, path=f'{c}_survival.png')
            all_ftau += [f_tau[-1, :]]
            all_tau += [tau[-1, :]]
            all_stau += [s_tau[-1,:]]
    
            ##  2D clustering
            step_dwell = np.array([step, dwell]).T
            # plt.plot(data_g, data_p, 'o')
            EM_gp = EM(step_dwell, dim=2)
            # n_components_gp = EM_gp.opt_components(tolerance=1e-3, mode='GP', criteria='BIC', figure=False)
    
            f1, m1, s1, f2, tau1 = EM_gp.GPEM(n_components=max(n_components_g,n_components_p), tolerance=1e-2, rand_init=True)
            # para = [f1[-1].ravel(), m1[-1].ravel(), s1[-1].ravel(), f2[-1].ravel(), tau1[-1].ravel()]
            # labels, data_cluster = EM_gp.predict(step_dwell, function=ln_gau_exp_pdf, paras=para)
            EM_gp.plot_gp_contour(save=True, path=f'{c}_2D.png')
    
    
            all_2d_f += [f1[-1,:]]
            all_2d_m += [m1[-1,:]]
            all_2d_s += [s1[-1,:]]
            all_2d_tau += [tau1[-1,:]]
    
            result = np.ones((5,4))*(-99)
            nr = len(all_2d_f[-1])
            result[:nr, 0] = all_2d_f[-1]
            result[:nr, 1] = all_2d_m[-1]
            result[:nr, 2] = all_2d_s[-1]
            result[:nr, 3] = all_2d_tau[-1]
            results += [result[result!=-99].reshape(nr,4)]
        all_results += [results]
