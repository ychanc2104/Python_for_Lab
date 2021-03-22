
from EM_Algorithm.gen_poisson import gen_poisson
import numpy as np
from EM_Algorithm.EM import EM

if __name__ == "__main__":
    ##  produce data

    data = gen_poisson(tau=[0.5, 4, 10], n_sample=[400]*3)
    n_sample = len(data)
    tolerance = 1e-3
    ##  fit Poisson EM
    EM_p = EM(data)
    # n_components = 2
    n_components_p = EM_p.opt_components(tolerance=tolerance, mode='PEM', criteria='AIC', figure=True)
    f, tau, s, labels, data_cluster = EM_p.PEM(n_components_p, tolerance)
    EM_p.plot_EM_results()
    EM_p.plot_fit_exp(xlim=[0,40])

