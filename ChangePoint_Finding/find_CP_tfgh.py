from basic.filter import MA
from ChangePoint_Finding.GradDescend_test_ChangePoint import *
from gen_assembly import nordata
from basic.select import select_file
import pandas as pd

path = select_file()
df = pd.read_excel(path, sheet_name="工作表1")
data_ori = np.array(df).T
data = nordata(data_ori)
n_data = data.shape[1]
# n_data = 4

p = np.reshape([4500, 5000]*n_data, (n_data,2))
save_fig = False ## save or not
n = 2 ## how many times you want to fit

## how many times you want to fit
for j in range(n):
    result = []
    for i in range(n_data):
        p_fit, converged, criteria_stop, Res = gradescent(data[i, :], p[i, :], tol=5e-2)
        p[i, :] = p_fit
        BM_initial_fit, BM_final_fit, velocity = plotresult(data_ori[i, :], p[i, :], dt=0.03)
        result = np.append(result, [BM_initial_fit, BM_final_fit, velocity])
        if j == n-1:
            plotresult(data_ori[i, :], p[i, :], dt=0.03, save=True, path=f'{i}.png')

result = np.reshape(result, (i+1, 3))
results = np.append(p, result, axis=1)

columns=['CP_1', 'CP_2', 'BM_initial_fit(nm)', 'BM_final_fit(nm)', 'velocity(nm/s)']
df_save = pd.DataFrame(data=results, columns=columns).to_excel('results.xlsx')
