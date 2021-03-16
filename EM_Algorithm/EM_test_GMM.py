
from EM_Algorithm.gen_gauss import gen_gauss
from EM_Algorithm.EM import EM

if __name__ == '__main__':
    n_sample = 2000
    data = gen_gauss(mean=[5,10], std=[2,2], n_sample=[n_sample]*2)

    ##  fit GMM
    EM = EM(data)
    n_components = 2
    f, m, s, labels, data_cluster = EM.GMM(n_components)
    EM.plot_EM_results()
    EM.plot_fit_gauss()

