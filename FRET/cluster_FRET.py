from basic.select import get_mat, get_files
from FRET.cluster_FRET_one import get_params, collect_params
import numpy as np
import pandas as pd
import random
import string
import os


def gen_random_code(n):
    digits = "".join([random.choice(string.digits) for _ in range(n)])
    chars = "".join([random.choice(string.ascii_letters) for _ in range(n)])
    return digits + chars

def reshape_results(*args):
    all_output = []
    for arg in args:
        n = []
        for x in arg:
            x = np.array(x,ndmin=1)
            n += [len(x)]
        n = max(n)
        output = np.zeros((len(args[0]), n))
        for i,x in enumerate(arg):
            x = np.array(x, ndmin=1)
            l = len(x)
            output[i,:l] = x
        all_output += [output]
    return all_output

def get_col(name,n):
    output = [f'component_{name}_{i+1}' for i in range(n)]
    return output

if __name__ == '__main__':


    n_component = 3 ## if None, auto-find n
    all_path = get_files('*.mat')
    # all_path = r'/home/hwligroup/Desktop/vbFRET_dwell time/*.mat'

    for path in all_path:
        name = os.path.split(path)[-1]
        data = get_mat(path)
        dwell = data['Transmat']

        EM_on, EM_off = [], []
        f_on, f_off = [], []
        tau_on, tau_off = [], []
        LLE_on, LLE_off = [], []

        m_shape = len(dwell)
        for i in range(1,len(dwell)):

            dwell_on = dwell[m_shape-i, m_shape-i-1]
            dwell_off = dwell[m_shape-i-1, m_shape-i]
            ## EM
            EM_p_on, f_tau_on, tau_i_on, s_tau_on, converged_p_on, ln_likelihood_on = get_params(dwell_on, n_component)
            EM_p_off, f_tau_off, tau_i_off, s_tau_off, converged_p_off, ln_likelihood_off = get_params(dwell_off)
            ## store results
            EM_on, EM_off, f_on, f_off, tau_on, tau_off, LLE_on, LLE_off = collect_params([EM_on, EM_p_on],
                                                                                          [EM_off, EM_p_off],
                                                                                          [f_on, f_tau_on],
                                                                                          [f_off, f_tau_off],
                                                                                          [tau_on, tau_i_on],
                                                                                          [tau_off, tau_i_off],
                                                                                          [LLE_on, ln_likelihood_on],
                                                                                          [LLE_off, ln_likelihood_off])

            # EM_p.plot_fit_exp(xlim=[0, 10])

        f_on, tau_on, f_off, tau_off, LLE_on, LLE_off = reshape_results(f_on, tau_on, f_off, tau_off, LLE_on, LLE_off)

        ## create dataframe
        df_f_on = pd.DataFrame(f_on, columns=get_col('f', f_on.shape[1]))
        df_tau_on = pd.DataFrame(tau_on, columns=get_col('tau', tau_on.shape[1]))
        df_LLE_on = pd.DataFrame(LLE_on, columns=get_col('LLE', LLE_on.shape[1]))
        df_on = pd.concat([df_f_on, df_tau_on, df_LLE_on], axis=1)

        df_f_off = pd.DataFrame(f_off, columns=get_col('f', f_off.shape[1]))
        df_tau_off = pd.DataFrame(tau_off, columns=get_col('tau', tau_off.shape[1]))
        df_LLE_off = pd.DataFrame(LLE_off, columns=get_col('LLE', LLE_off.shape[1]))
        df_off = pd.concat([df_f_off, df_tau_off, df_LLE_off], axis=1)
        df = [df_on, df_off]
        sheet_names = ['on', 'off']

        ## save to disk
        writer = pd.ExcelWriter( f'{gen_random_code(3)}_{name}_EM_results.xlsx')
        for i in range(2):
            df[i].to_excel(writer, sheet_name=sheet_names[i], index=True)
        writer.save()