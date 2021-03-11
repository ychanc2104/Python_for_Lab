from gen_gauss import gen_gauss, binning
from EM_stepsize import GMM, plot_EM_results, plot_fit_gauss


mean = 5
std = 3
n_sample = 200
data = gen_gauss(mean=[5, 15, 25, 12], std=[2, 5, 3, 4], n_sample=[n_sample]*4)

##  fit GMM
n_components = 4
m_save, s_save, f_save, label, data_cluster = GMM(data, n_components, tolerance=10e-3)
m = m_save[-1, :]
s = s_save[-1, :]
f = f_save[-1, :]

##  plot EM results(mean, std, fractio with iteration)
plot_EM_results(m_save, s_save, f_save)

##  plot data histogram and its gaussian EM (GMM) results
plot_fit_gauss(data, f, m, s)

