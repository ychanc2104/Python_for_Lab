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

### import data
path = select_file()
df = pd.read_excel(path)
data = np.array(df)
n_traces = int(df.shape[1]/4)
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
t = np.arange(0.05, (min(t_end)-0.05)/2, 0.01)
varX_all = []
semX_all = []
velocity_all = []
for i,signal in enumerate(signals):
    print(f'analyzing variance of trace #{i}')
    fs = Fs[i]
    varX = []
    semX = []
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
    varX_all += [np.array(varX)]
    semX_all += [np.array(semX)]
    velocity_all += [v]

### fit all time vs variance(t)
slope_all = []
intercept_all = []
points_tofit = 5
t_fit = t[0:points_tofit]
plt.figure()
for varX, semX in zip(varX_all, semX_all):
    print(f'analyzing slope of time-variance #{i}')
    varX_fit = varX[0:points_tofit]
    slope, intercept = L_fit(t_fit, varX_fit)
    slope_all += [slope]
    intercept_all += [intercept]
    # plt.figure()
    plt.errorbar(t, varX, yerr=semX, color='dodgerblue', marker='o', ls='--', capsize=5, capthick=1, ecolor='black')
    plt.plot(t, linear_eq(t, slope, intercept), 'r--')
    plt.xlabel('Time (s)', fontsize=22)
    plt.ylabel('Variance ($\mathregular{count^2}$)', fontsize=22)

## parameters for average of all fitting slopes and intercepts
slope = np.mean(slope_all)
intercept = np.mean(intercept_all)


### fit a averaged-varX
varX_1 = np.mean(np.array(varX_all), axis=0)
semX_1 = np.sqrt(np.sum(np.array(semX_all)**2, axis=0)/len(semX_all))
varX_1_fit = varX_1[0:points_tofit]
## parameters for fitting average of all time-variance traces
slope_1, intercept_1 = L_fit(t_fit, varX_1_fit)
plt.figure()
plt.errorbar(t, varX_1, yerr=semX_1, color='dodgerblue', marker='o', ls='--', capsize=5, capthick=1, ecolor='black')
plt.plot(t, linear_eq(t, slope_1, intercept_1), 'r--')
plt.xlabel('Time (s)', fontsize=22)
plt.ylabel('Variance ($\mathregular{count^2}$)', fontsize=22)

### analyze final results
"""
step = slope/velocity
k = velocity/step = velocity^2/slope
"""
step_fit_avg = np.mean([slope/v for slope, v in zip(slope_all, velocity_all)])
k_fit_avg = np.mean([v**2/slope for slope, v in zip(slope_all, velocity_all)])

step_avg_fit = slope_1/np.mean(velocity_all)
k_avg_fit = np.mean(velocity_all)/step_avg_fit

print(f'step size of fitting all traces and averaging is {step_fit_avg}')
print(f'k of fitting all traces and average is {k_fit_avg}')
print(f'step size of averaging all traces and fitting is {step_avg_fit}')
print(f'k of averaging all traces and fitting is {k_avg_fit}')

