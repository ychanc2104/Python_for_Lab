
from matplotlib import rcParams
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial"]
from EM_Algorithm.EM_stepsize import GMM, plot_EM_results, plot_fit_gauss
from EM_Algorithm.EM_test_Poisson import exp_EM, plot_EM_results_exp, plot_fit_pdf, exp_pdf
from basic.select import get_files, select_folder
import scipy.io as sio
import numpy as np

conc = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]

path_folder = select_folder()
all_step = []
all_dwell = []
n_components_g = 2
n_components_e = 2

centers = []
stds = []
fractions = []

all_tau = []
all_ftau = []
for c in conc:
    path_data = get_files(f'S5S1_{c}*.mat', dialog=False, path_folder=path_folder)
    step = []
    dwell = []
    for path in path_data:
        data = sio.loadmat(path)
        step = np.append(step, data['step'])
        dwell = np.append(dwell, [data['dwell']])
    m_save, s_save, f_save, label, data_cluster = GMM(step, n_components_g, tolerance=10e-5)
    m = m_save[-1,:]
    s = s_save[-1,:]
    f = f_save[-1,:]
    ##  plot EM results(mean, std, fractio with iteration)
    plot_EM_results(m_save, s_save, f_save)


    ### fit dwell
    ##  plot data histogram and its gaussian EM (GMM) results
    plot_fit_gauss(step, f, m, s)
    centers += [m]
    stds += [s]
    fractions += [f]

    ##  fit EM
    f_save, tau_save = exp_EM(dwell, n_components_e, tolerance=10e-5)

    ##  plot EM results(mean, std, fractio with iteration)
    plot_EM_results_exp([f_save, tau_save], ['fraction', 'dwell time'])

    ##  plot data histogram and its gaussian EM (GMM) results
    plot_fit_pdf(dwell, exp_pdf, [f_save[-1,:], tau_save[-1,:]])

    all_tau += [tau_save[-1,:]]
    all_ftau += [f_save[-1,:]]

    all_step += [step]
    all_dwell += [dwell]


