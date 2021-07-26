
from basic.select import get_files, select_folder
from FRET.cluster_FRET import get_col, gen_random_code
import scipy.io as sio
import pandas as pd
from EM_Algorithm.EM import *
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial"]
rcParams.update({'font.size': 12})

def remove_steps(step, dwell, criteria):
    booleans = step < criteria
    return step[booleans], dwell[booleans]


def collect_all(path_folder, conc):
    step_all = []
    dwell_all = []
    for c in conc:
        path_data = get_files(f'*{c}*.mat', dialog=False, path_folder=path_folder)
        step = []
        dwell = []
        for path in path_data:
            data = sio.loadmat(path)
            step = np.append(step, data['step'])
            dwell = np.append(dwell, [data['dwell']])
            step, dwell = remove_steps(step, dwell, criteria=15)
        step_all = np.append(step_all, step)
        dwell_all = np.append(dwell_all, dwell)
    return step_all, dwell_all


if __name__ == '__main__':
    all_gauss = []
    all_survival = []
    all_results = []

    n_sample = []
    # conc = ['0.0', '0.5', '1.0', '1.5', '2.0', '3.0']  ## S5S1
    # conc = [1.0, 1.2, 1.5, 1.8, 2.0, 3.0, 4.0] ## m51 only
    # conc = ['0.10', '0.20', '0.25', '0.50', '0.70', '0.80', '1.10', '1.20', '2.00']  ## EcRecA
    conc = ['0.10']
    # path_folder = select_folder()
    # path_folder = r'/home/hwligroup/Desktop/Data/step-dwell time/m51'
    # path_folder = r'/home/hwligroup/Desktop/Data/step-dwell time/m51 + mSS'
    # path_folder = r'C:\Users\pine\Desktop\Data\step-dwell time\EcRecA'
    path_folder = r'C:\Users\pine\Desktop\Data\step-dwell time\m51 + mSS'
    # path_folder = r'C:\Users\pine\Desktop\Data\step-dwell time\m51 + mSS - v2(0619)'
    # path_folder = r'C:\Users\pine\Desktop\Data\step-dwell time\m51 + mSS - v3(0621)'
    step, dwell = collect_all(path_folder, conc)

    ## get poisson EM results
    EM_p = EM(dwell)
    # n_components_p = EM_p.opt_components_iter(tolerance=1e-2, mode='PEM', criteria='BIC', figure=False)
    f_tau, tau, s_tau, converged_p, LLE = EM_p.PEM(2)

    ##  2D clustering
    step_dwell = np.array([step, dwell]).T
    EM_gp = EM(step_dwell, dim=2)
    # opt_components = EM_gp.opt_components_iter(tolerance=1e-2, mode='GPEM', criteria='BIC', figure=False)
    # f1, m1, s1, tau1, converged_gp, LLE_gp = EM_gp.GPEM(n_components=2, tolerance=1e-2, rand_init=True)
    f1, m1, s1, tau1, converged_gp, LLE_gp = EM_gp.GPEM_set(n_components=2, m_set=np.array([4.59,8.13]), tolerance=1e-2, rand_init=True)

    ##  plot figure
    # EM_g.plot_fit_gauss(scatter=False, xlim=[0, 16], save=False, path=f'{c}_gauss.png', figsize=(7,2))
    # EM_p.plot_fit_exp(xlim=[0, 10], save=False, path=f'{c}_survival.png', figsize=(7,2))
    para = EM_gp.para_final
    EM_p.plot_fit_exp(xlim=[0, 10], save=True, path=f'EcRecA_{conc[0]}_survival.png', figsize=(7,2),
                              para=[para[0]]+[para[3]], fontsize=12, remove_ytick=True, ylabel='', line_width=2.5)

    # EM_gp.plot_gp_contour(xlim=[0, 20], ylim=[0, 10.], save=True, path=f'{c}_2D.png')
    # EM_gp.plot_gp_surface()
    # EM_gp.plot_gp_contour_2hist(xlim=[0, 12], ylim=[0, 10], bins_x=np.log2(len(step)).astype('int'),
    #                             save=False, path=f'SS_all_scatter.png')
    EM_gp.plot_gp_contour_2hist(xlim=[0, 12], ylim=[0, 10], bins_x=np.log2(len(step)).astype('int'),
                                save=True, path=f'EcRecA_{conc[0]}_scatter.png', figsize=(6, 5))

    all_results += [np.array([f1, m1, s1, tau1, converged_gp, LLE_gp]).T]
    print(all_results)

    # df_f = pd.DataFrame(f1, columns=['f'])
    # df_m = pd.DataFrame(m1, columns=['center'])
    # df_s = pd.DataFrame(s1, columns=['std'])
    # df_tau = pd.DataFrame(tau1, columns=['tau'])
    # df_LLE = pd.DataFrame(LLE_gp, columns=['LLE'])
    # df = pd.concat([df_f, df_m, df_s, df_tau, df_LLE], axis=1)
    #
    # ## save to disk
    # writer = pd.ExcelWriter(f'{gen_random_code(3)}_EM_results.xlsx')
    # df.to_excel(writer, sheet_name='EM', index=True)
    # writer.save()




