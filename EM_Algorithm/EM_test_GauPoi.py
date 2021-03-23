from EM_Algorithm.gen_gauss import gen_gauss
from EM_Algorithm.gen_poisson import gen_poisson
from EM_Algorithm.EM import *

if __name__ == '__main__':
    n_sample = 500
    data_g = gen_gauss(mean=[4, 6], std=[2,2], n_sample=[n_sample]*2)
    data_p = gen_poisson(tau=[1, 5], n_sample=[n_sample]*2)
    data = np.array([data_g, data_p]).T
    # plt.plot(data_g, data_p, 'o')
    EM_gp = EM(data, dim=2)
    f1, m, s1, f2, tau = EM_gp.GPEM(2, tolerance=1e-10)
    para = [f1[-1].ravel(), m[-1].ravel(), s1[-1].ravel(), f2[-1].ravel(), tau[-1].ravel()]
    labels, data_cluster = EM_gp.predict(data, function=ln_gau_exp_pdf, paras=para)

    plt.figure()
    for x in data_cluster:
        plt.plot(x[:,0], x[:,1],'o')
    # plt.plot(data_g, data_p, 'o', markersize=2)

