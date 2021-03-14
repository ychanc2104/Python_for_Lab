# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 13:48:42 2021
"""
from matplotlib import rcParams
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial"]

from basic.binning import binning
import numpy as np
import matplotlib.pyplot as plt
import math

##  args: list of parameters, x: np array
def oneD_gaussian(x, args):
    # x: (n,)
    # args: (k), args[0,1,2...,k-1]: (m)
    # output: (k,n)
    f = np.array(args[0])
    xm = np.array(args[1])
    s = np.array(args[2])
    y = []
    for f, xm, s in zip(f, xm, s):
        y += [f*1/s/np.sqrt(2*math.pi)*np.exp(-(x-xm)**2/2/s**2)]
    return np.array(y)

def log_oneD_gaussian(x, args):
    f = np.array(args[0])
    xm = np.array(args[1])
    s = np.array(args[2])
    lny = []
    for f, xm, s in zip(f, xm, s):
        lny += [np.log(f) - np.log(s*np.sqrt(2*math.pi)) - (x-xm)**2/2/s**2]
    return np.array(lny)

##  plot data histogram and its gaussian EM (GMM) results
def plot_fit_gauss(data, f, m, s):
    n_components = len(m)
    bin_number = np.log2(len(data)).astype('int') + 1
    pd, center = binning(data, bin_number) # plot histogram
    data_std_new = np.std(data, ddof=1)
    x = np.arange(0, max(data)+data_std_new, 0.05)
    y_fit = oneD_gaussian(x, [f, m, s])
    for i in range(n_components):
        plt.plot(x, y_fit[i,:], '-')
    plt.plot(x, sum(y_fit), '-')
    plt.xlabel('step size (nm)', fontsize=15)
    plt.ylabel('probability density (1/$\mathregular{nm^2}$)', fontsize=15)

def GMM(data, n_components, tolerance=10e-5):
    ##  initialize EM parameters
    m, s, f = init_GMM(data, n_components=n_components)
    j = 0
    m_save = []
    s_save = []
    f_save = []
    improvement = 10
    while (j < 100 or improvement > tolerance) and j < 5000:
        m_save, s_save, f_save = collect_EM_results([m_save, m], [s_save, s], [f_save, f])
        prior_prob = weighting(data, m, s, f)
        m, s, f = update_m_s_f(prior_prob, data)
        improvement = cal_improvement([m_save[-n_components:], m], [s_save[-n_components:], s], [f_save[-n_components:], f])
        j += 1
    m_save, s_save, f_save = collect_EM_results([m_save, m], [s_save, s], [f_save, f])
    m_save, s_save, f_save = reshape_all(m_save, s_save, f_save, n_components=n_components)
    label = get_label(data, prior_prob)
    data_cluster = [data[label == i] for i in range(n_components)]
    ln_likelihood = [sum(log_oneD_gaussian(data, args=[f, m, s])[i]) for i,data in enumerate(data_cluster)]
    ln_likelihood = sum(ln_likelihood)
    BIC = -2*ln_likelihood + (n_components*3-1)*np.log(len(data))
    return m_save, s_save, f_save, label, data_cluster, BIC

    ##  initialize mean, std and fraction of gaussian
def init_GMM(data, n_components):
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    m = np.arange(mean-1.5*std, mean+1.5*std+1, 3*std/(n_components-1))
    s = np.array([std]*n_components)
    f = np.ones(n_components) / n_components
    return m, s, f

##  calculate the probability belonging to each cluster, (m,s)
def weighting(data, m, s, f):
    p = oneD_gaussian(data, args=[f, m, s])
    prior_prob = p / sum(p)
    return prior_prob

##  update mean, std and fraction using matrix multiplication, (n_feture, n_sample) * (n_sample, 1) = (n_feture, 1)
def update_m_s_f(likelihood, data):
    n_sample = len(data)
    m = np.matmul(likelihood, data) / np.sum(likelihood, axis=1)
    s = np.sqrt(np.matmul(likelihood, data ** 2)/np.sum(likelihood, axis=1)-m**2)
    f = np.sum(likelihood, axis=1) / n_sample
    return m, s, f

def plot_EM_result(result, ax, xlabel='iteration', ylabel='value'):
    n_feature = result.shape[1]
    iteration = result.shape[0]
    for i in range(n_feature):
        ax.plot(np.arange(0, iteration), result[:, i], '-o')
    ax.set_ylabel(f'{ylabel}', fontsize=15)

def plot_EM_results(m, s, f):
    fig, axs = plt.subplots(3, sharex=True)
    axs[2].set_xlabel('iteration', fontsize=15)
    plot_EM_result(m, axs[0], ylabel='mean')
    plot_EM_result(s, axs[1], ylabel='std')
    plot_EM_result(f, axs[2], ylabel='fraction')

def cal_improvement(*args):
    ##  arg: [m_old, m_new]
    improvement = 0
    for arg in args:
        dx = abs(arg[0] - arg[1])
        improvement = max(np.append(improvement, dx)) # take max of all args diff
    return improvement

def collect_EM_results(*args):
    results = []
    for arg in args:
        results += [np.append(arg[0], arg[1])]
    return results

def get_label(data, likelihood):
    label = []
    for i, x in enumerate(data):
        p = likelihood[:, i]
        label += [np.argmax(p)]
    label = np.array(label)
    return label

def reshape_all(*args, n_components):
    results = []
    n_sample = int(len(args[0])/n_components)
    for arg in args:
        results += [np.reshape(arg, (n_sample, n_components))]
    return results

if __name__ == '__main__':
    ##  import data
    # data = np.array([179, 165, 175, 185, 158, 190])
    data = np.array([4.35025209568627,6.02028403216761,8.74013549119374,8.79569063382595,6.30754174712644,6.17940619409282,2.56770170740741,4.09325603111111,4.05575733490197,4.98339941176471,3.34095459770115,5.00863758919531,7.38938795612111,7.39868092243186,4.40006851851852,6.27209499579082,5.35435634795918,4.18512907894737,3.28353755685509,9.16097426042396,5.63977660377359,7.12999459684424,6.53892494674741,8.72490313997478,5.12361121794871,9.96075507575757,4.63959454545455,4.05804113082041,2.32118111701031,2.96073818298969,5.73656437500000,5.53746076388889,2.92860277777778,2.94117303921568,2.96609860999415,4.43907202918790,9.15801579831469,6.47145942681818,5.82521718750000,11.5910488541667,2.35736168382353,4.33874518676471,2.17904692307692,2.42293252136752,2.27889488888889,3.40683852380952,3.38829304029305,4.11920620782726,9.00506035314685,3.59073716186253,4.36922064562410,4.23581264705882,3.24299982758621,8.83586320583333,8.48158618461538,5.01307128205129,13.6414206250000,3.53869546764706,7.39194575000000,5.67960400000000,2.59289492307693,5.95084875259875,10.1461682668816,8.17695539376624,6.04041297208539,4.79999806687566,5.97587668478261,6.90407480331263,5.22437378151260,11.3061579696395,8.34094021020408,9.80657928571429,4.26540867346939,4.72541359925788,9.35106106060606,6.61003634751770,9.15166902061106,10.5560535551465,7.02705234811166,7.53111469449486,2.77092562252404,5.70248472820513,4.30679525274726,3.54356904433497,3.43371724137931,8.15848085106383,6.76741071840000,3.71760370833333,4.26958538690476,2.34370449735451,5.34461455026455,3.56330604597701,3.48403185185185,2.59677200779727,3.66513947368421,3.30535000000000,9.52206230491804,6.18010657101449,5.63024459308807,5.41033917901669,4.67241280233528,6.24400914966036,9.78777438342209,7.39591117084155,10.1914972469325,9.85872378424658,8.78486689757160,9.91476024751675,8.97093417312407,5.61877513027866,5.56929660188998,5.96719300884956,16.2050459773942,7.68322756344598,4.18777229980620,5.36938336309524,4.11505008597883,6.76934169346979,5.79203198738926,4.00763740289414,5.21532623076923,7.63250354545455,4.65773116883117,9.33270112781954,5.66747811551783,5.13917333165323,5.32319821082746,4.34257265662943,3.42493850074964,5.29084105590061,6.85445320197043,4.05557226386807,6.31409484855378,7.27299129950629,3.72318544674325,5.73032427951518,7.15471772951794,4.40682487276951,7.49431772463308,6.80203304935065,3.37185547878789,7.39921446334311,6.22458818181819,4.66357500000000,6.62360071428572,7.74363246753247,2.66032070707072,8.15780895927173,5.02469860657895,16.0765605451128,6.29293597402597,4.81025136363638,3.09694347130682,4.87115098181818,6.53759592727273]
                    )
    ##  fit GMM
    n_components = 1
    m_save, s_save, f_save, label, data_cluster, BIC = GMM(data, n_components, tolerance=10e-3)
    m = m_save[-1,:]
    s = s_save[-1,:]
    f = f_save[-1,:]

    ##  plot EM results(mean, std, fractio with iteration)
    plot_EM_results(m_save, s_save, f_save)

    ##  plot data histogram and its gaussian EM (GMM) results
    plot_fit_gauss(data, f, m, s)