
from ChangePoint_Finding.GradDescend_test_ChangePoint import *
import pandas as pd
from basic.select import select_file



path = select_file()
df = pd.read_excel(path)

data_ori = np.array(df)
data_ori = data_ori[data_ori > 0.1]
data = nordata(data_ori)
p_initial = np.array([5000,5500])
p, converged, criteria_stop = gradescent(data, p_initial, tol=1e-2)

BM_initial_fit, BM_final_fit, velocity = plotresult(data_ori, p)

plt.figure()
plt.plot(criteria_stop)