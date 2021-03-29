from matplotlib import rcParams
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial"]
rcParams.update({'font.size': 18})
import pandas as pd
from basic.select import select_file
import numpy as np
import statistics as stat
import matplotlib.pyplot as plt
from basic.fitting import linear_eq, L_fit
from basic.file_io import save_img

### import data
path = select_file()
df = pd.read_excel(path)
data = np.array(df.dropna(axis='columns', how='all'))
n_traces = int(data.shape[1]/4)
signals = []
dt = []
t_end = [] # ending time
Fs = []
for i in range(n_traces):
    ##  select and remove nan data
    signal = data[:, 1+4*i]
    signals += [signal[~np.isnan(signal)]]
    dt += [data[1, 0+4*i] - data[0, 0+4*i]]
    t_end += [dt[-1]*len(signals[-1])]
    Fs += [1/dt[-1]]

### calculate variance
t = np.arange(0.05, (min(t_end)-0.05)/2, 0.04)
varX_all = []
semX_all = []
velocity_all = []
xm_all = []
for i,signal in enumerate(signals):
    print(f'analyzing variance of trace #{i}')
    fs = Fs[i]
    varX = []
    semX = []
    xm = []
    t_trace = np.arange(0, len(signal)) / fs
    v, x0 = L_fit(t_trace, signal)
    for ti in t:
        n_interval = int(np.floor(ti*fs)) ## time diff for calculating variance
        n_row = int(np.floor(len(signal)/n_interval)) ##
        n_samples = n_interval*n_row
        x = signal[0:n_samples]
        x_reshape = x.reshape(n_row, n_interval)
        x_diff = np.diff(x_reshape, axis=0).ravel()
        varX += [stat.variance(x_diff)] ## sample variance
        semX += [np.std(varX, ddof=1)]
        xm += [np.mean(x_diff)]
    varX_all += [np.array(varX)]
    semX_all += [np.array(semX)]
    velocity_all += [v]
    xm_all += [np.array(xm)]

### fit all time vs variance(t)
slope_all = []
intercept_all = []
randomness_fit_avg_all = []
points_tofit = 15
t_fit = t[0:points_tofit]
# t_fit = t[points_tofit:]

plt.figure()
i = 0
for varX, semX, xm, velocity in zip(varX_all, semX_all, xm_all, velocity_all):
    print(f'analyzing slope of time-variance #{i}')
    ##  fit initial
    varX_fit = varX[0:points_tofit]
    xm_fit = xm[0:points_tofit]
    ## fit last section
    # varX_fit = varX[points_tofit:]
    # xm_fit = xm[points_tofit:]
    
    slope, intercept = L_fit(t_fit, varX_fit)
    slope_all += [slope]
    intercept_all += [intercept]
    randomness_fit_avg_all += [(varX_fit[-1] - intercept)/xm_fit[-1] ]
    # plt.figure()
    plt.errorbar(t, varX, yerr=semX, color='dodgerblue', marker='o', ls='--', capsize=5, capthick=1, ecolor='black')
    plt.plot(t, linear_eq(t, slope, intercept), 'r--')
    plt.xlabel('Time (s)', fontsize=22)
    plt.ylabel('Variance ($\mathregular{count^2}$)', fontsize=22)
    i += 1

slope_all = np.array(slope_all)
intercept_all = np.array(intercept_all)
### parameters for average of all fitting slopes and intercepts
s = np.std(slope_all)
m = np.mean(slope_all)

booleans = (slope_all > m-s) & (slope_all < m+s)
slope_all = slope_all[booleans]
intercept_all = intercept_all[booleans]
slope = np.mean(slope_all)
intercept = np.mean(intercept_all)
##  remove outlier



### fit a averaged-varX
varX_1 = np.mean(np.array(varX_all), axis=0)
semX_1 = np.sqrt(np.sum(np.array(semX_all)**2, axis=0)/len(semX_all))
xm_1 = np.mean(np.array(xm_all), axis=0)
##  fit initial
varX_1_fit = varX_1[0:points_tofit]
xm_1_fit = xm_1[0:points_tofit]
## fit last section
# varX_1_fit = varX_1[points_tofit:]
# xm_1_fit = xm_1[points_tofit:]

## parameters for fitting average of all time-variance traces
slope_1, intercept_1 = L_fit(t_fit, varX_1_fit)
fig, ax = plt.subplots(figsize=(10,8))
ax.errorbar(t, varX_1, yerr=semX_1, color='dodgerblue', marker='o', ls='--', capsize=5, capthick=1, ecolor='black')
ax.plot(t, linear_eq(t, slope_1, intercept_1), 'r--')
ax.set_xlabel('Time (s)', fontsize=22)
ax.set_ylabel('Variance ($\mathregular{count^2}$)', fontsize=22)
save_img(fig, 'VA_2.0.png')

# randomness_avg_fit = (varX_1_fit[-1] - intercept_1)/()
### analyze final results
"""
step = slope/velocity
k = velocity/step = velocity^2/slope
pre_randomness, r = (var-intcept)/xm
"""
step_fit_avg = np.mean([slope/v for slope, v in zip(slope_all, velocity_all)])
k_fit_avg = np.mean([v**2/slope for slope, v in zip(slope_all, velocity_all)])
r_fit_avg = np.mean(randomness_fit_avg_all)

step_avg_fit = slope_1/np.mean(velocity_all)
k_avg_fit = np.mean(velocity_all)/step_avg_fit
r_avg_fit = (varX_1_fit[-1] - intercept_1)/(xm_1_fit[-1])

print(f'step size of fitting all traces and averaging is {step_fit_avg}')
print(f'k of fitting all traces and average is {k_fit_avg}')
print(f'randomness of fitting all traces and average is {r_fit_avg}')

print(f'step size of averaging all traces and fitting is {step_avg_fit}')
print(f'k of averaging all traces and fitting is {k_avg_fit}')
print(f'randomness of averaging all traces and fitting i {r_avg_fit}')

