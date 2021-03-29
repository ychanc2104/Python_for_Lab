
from EM_Algorithm.gen_gauss import gen_gauss
from EM_Algorithm.EM import *

if __name__ == '__main__':
    n_sample = 200
    data = gen_gauss(mean=[10,13], std=[2,2], n_sample=[n_sample,500])
    data = data.reshape(-1,1)
    n_sample = len(data)
    ##  fit GMM
    EMg = EM(data)
    n_components = 2
    opt_components = EMg.opt_components(tolerance=1e-2, mode='GMM', criteria='AIC', figure=False)

    f, m, s, labels, data_cluster = EMg.GMM(opt_components, tolerance=1e-2, rand_init=False)
    EMg.plot_EM_results()
    EMg.plot_fit_gauss(scatter='True')

    # f2, m2, s2, labels2, data_cluster2 = EM.skGMM(n_components)
    # EM.plot_fit_gauss()




