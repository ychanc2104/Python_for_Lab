from basic.select import get_mat
from EM_Algorithm.EM import EM
import os

def get_params(dwell, n_component=None):
    EM_p = EM(dwell)
    if n_component == None:
        n_components_p = EM_p.opt_components_iter(tolerance=1e-2, mode='PEM', criteria='BIC', figure=False)
    else:
        n_components_p = n_component
    f_tau, tau, s_tau, converged_p, ln_likelihood = EM_p.PEM(n_components_p, rand_init=True)
    return EM_p, f_tau, tau, s_tau, converged_p, ln_likelihood

def collect_params(*args):
    output = []
    for arg in args:
        arg[0] += [arg[1]]
        output += [arg[0]]
    return output




if __name__ == '__main__':

    # path = select_file()
    path = r'/home/hwligroup/Desktop/vbFRET_dwell time/SS/Transmat_Dmc1+1SS.mat'
    state = [2,3]
    n_component = 2 ## if n is None, auto-find n

    ## parse data
    name = os.path.split(path)[-1]
    data = get_mat(path)
    dwell_matrix = data['Transmat']

    EM_all = []
    f_all = []
    tau_all = []
    LLE_all = []
    m_shape = len(dwell_matrix)

    dwell = dwell_matrix[state[0], state[1]]
    ## EM
    EM_p, f_tau, tau_i, s_tau, converged_p, ln_likelihood = get_params(dwell, n_component=n_component)
    ## store results
    EM_all, f_all, tau_all, LLE_all = collect_params([EM_all, EM_p],
                                                     [f_all, f_tau],
                                                     [tau_all, tau_i],
                                                     [LLE_all, ln_likelihood])

    EM_p.plot_fit_exp(xlim=[0, 10], save=True, path='output.png', xlabel='dwell time (s)', ylabel='survival', figsize=(10,10))
    EM_p.plot_EM_results()

    print(f'fraction: {f_all}\n'
          f'tau: {tau_all}\n'
          f'log likelihood: {LLE_all}')
