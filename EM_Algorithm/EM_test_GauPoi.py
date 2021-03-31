from EM_Algorithm.gen_gauss import gen_gauss
from EM_Algorithm.gen_poisson import gen_poisson
from EM_Algorithm.EM import *

if __name__ == '__main__':
    n_sample = 500
    data_g = gen_gauss(mean=[4, 5], std=[1.5, 2], n_sample=[n_sample, 1000])
    data_p = gen_poisson(tau=[3, 1], n_sample=[n_sample, 1000])
    data = np.array([data_g, data_p]).T
    EM_gp = EM(data, dim=2)
    f1, m, s1, tau = EM_gp.GPEM(2, tolerance=1e-2, rand_init=False)
    EM_gp.plot_EM_results()
    EM_gp.plot_gp_contour()
    opt_components = EM_gp.opt_components(tolerance=1e-2, mode='GPEM')

    prior_prob = EM_gp.prior_prob

