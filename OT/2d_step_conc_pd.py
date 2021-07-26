from basic.math_fn import gauss
from basic.file_io import save_img
from matplotlib import rcParams
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial"]
rcParams.update({'font.size': 12})
import numpy as np
import matplotlib.pyplot as plt

f_all = [1.00,0.45,0.55,0.68,0.32,0.65,0.35,0.99,0.010]
m_all = [8.10,4.55,8.71,4.61,7.85,4.30,7.52,4.92,14.13]
s_all = [3.08,1.02,2.35,1.03,2.32,1.15,2.61,1.69,1.110]
c_all = [0.00,0.50,0.50,1.50,1.50,2.00,2.00,3.00,3.000]
fig_n = [0   ,1   ,1   ,2   ,2   ,3   ,3   ,4   ,4    ]
colors= [ 'b', 'g', 'b', 'g', 'b', 'g', 'b', 'g', 'b' ]

mean =  [8.1,6.84,5.66,5.43,5.04]
c_mean = [0, 0.50,1.50,2.00,3.00] ## conc, remove 1.0
err = [0.48, 0.12,0.14,0.16,0.19] ## SEM

fig = plt.figure(figsize=(5,13))
gs = fig.add_gridspec(5, hspace=0)
axs = gs.subplots(sharex=True, sharey=True)

x = np.linspace(0, 12, 200)
for i in range(len(f_all)):
    y = f_all[i]*gauss(x, xm=m_all[i], s=s_all[i])
    axs[fig_n[i]].plot(x, y, color=colors[i])
    axs[fig_n[i]].spines[:].set_linewidth('1.5')  ## xy, axis width
    axs[fig_n[i]].tick_params(width=1.5)  ## tick width

for j in range(5): ## five figures

    axs[j].errorbar(mean[j], 0.03, xerr=err[j], fmt='none', ecolor='r',
                    color='r', elinewidth=3, capsize=5)
    axs[j].scatter(mean[j], 0.03, s=30, color='r')
    axs[j].set_xlabel('Step-size (count)')
    axs[j].set_xlim(0, 12)
    axs[fig_n[i]].spines[:].set_linewidth('1.5') ## xy, axis width
    axs[fig_n[i]].tick_params(width=1.5) ## tick width

save_img(fig, 'output.png')