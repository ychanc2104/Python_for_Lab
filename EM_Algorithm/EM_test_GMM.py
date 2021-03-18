
from EM_Algorithm.gen_gauss import gen_gauss
from EM_Algorithm.EM import *
from sklearn.mixture import GaussianMixture
import numpy as np

if __name__ == '__main__':
    n_sample = 2000
    data = gen_gauss(mean=[6,10], std=[2,2], n_sample=[n_sample]*2)
    data = data.reshape(-1,1)
    n_sample = len(data)
    ##  fit GMM
    EMg = EM(data)
    n_components = 4
    f, m, s, labels, data_cluster = EMg.GMM(n_components, tolerance=1e-2)
    EMg.plot_EM_results()
    EMg.plot_fit_gauss()


    # f2, m2, s2, labels2, data_cluster2 = EM.skGMM(n_components)
    # EM.plot_fit_gauss()
    opt_components = EMg.opt_components(tolerance=1e-2, mode='GMM', criteria='AIC', figure=True)




