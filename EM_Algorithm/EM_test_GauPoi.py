import matplotlib.pyplot as plt

from EM_Algorithm.gen_gauss import gen_gauss
from EM_Algorithm.gen_poisson import gen_poisson
from EM_Algorithm.EM import *

if __name__ == '__main__':
    fontsize = 12
    n_sample = 200
    m_set = [5, 10]
    s_set = [1, 2]
    tau_set = [0.5, 1]
    # m_set = [10]
    # s_set = [2]
    # tau_set = [1]
    figsize = (6,5)
    data_g = gen_gauss(mean=m_set, std=s_set, n_sample=[n_sample*2, n_sample])
    data_p = gen_poisson(tau=tau_set, n_sample=[n_sample*2, n_sample])
    # fig, ax = plt.subplots(figsize=figsize)
    # ax.plot(data_g, data_p, 'o')
    # ax.set_xlabel('step size (a.u.)', fontsize=fontsize)
    # ax.set_ylabel('dwell time (s)', fontsize=fontsize)
    EM_p = EM(data_p)
    f_tau, tau, s_tau, converged_p, LLE = EM_p.PEM(2)

    data = np.array([data_g, data_p]).T
    EM_gp = EM(data, dim=2)
    opt_components = EM_gp.opt_components_iter(tolerance=1e-2, mode='GPEM')
    f1, m, s1, tau, converged, LLE = EM_gp.GPEM(opt_components, tolerance=1e-2, rand_init=False)

    # EM_gp.plot_gp_contour(xlim=[0, 20], ylim=[0, 10.])
    # EM_gp.plot_gp_surface()
    EM_p.plot_fit_exp(xlim=[0, 10], figsize=(7,2), fontsize=12, remove_ytick=True, ylabel='', line_width=2.5)
    EM_gp.plot_gp_contour_2hist(xlim=[0, 20], ylim=[0, 10], figsize=figsize)

    prior_prob = EM_gp.prior_prob
    print(f'Converged result is {converged}')
    print(f'gauss fraction is {f1}')
    print(f'gauss center is {m}\n expected is {m_set}')
    print(f'gauss std is {s1}\n expected is {s_set}')
    print(f'dwell time is {tau}\n expected is {tau_set}')
