# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 13:29:55 2020

find two point to mininize square error of
BM before(const.) + one dropping line + BM after(const.)
loss function is un-differentiable
"""


##

from ChangePoint_Finding.gen_assembly import *
import pandas as pd
from basic.select import select_file

##
def slopecurve(data, p):
    t1 = int(p[0])
    t2 = int(p[1])
    c1 = np.nanmean(data[0:t1])
    c2 = np.nanmean(data[t2:])
    curve = np.linspace(c1, c2, abs(t2 - t1))

    return curve

def lossfun(data, p):
    # p: [t1, t2]
    t1 = int(p[0])
    t2 = int(p[1])
    c1 = np.mean(data[0:t1])
    c2 = np.mean(data[t2:])
    sse1 = sum((data[0:t1] - c1)**2)
    sse2 = sum((data[t2:] - c2)**2)
    sse3 = sum((data[t1:t2] - slopecurve(data, p))**2)
    return sse1 + sse2 + sse3

def gradL(data, p):
    grad_t1 = lossfun(data, [p[0]+1, p[1]]) - lossfun(data, p)
    grad_t2 = lossfun(data, [p[0], p[1]+1]) - lossfun(data, p)
    return np.array([grad_t1, grad_t2])

def gradescent(data, p_initial, tol=1e-2):
    p = p_initial
    l = 3 # learning rate
    Method = 'Momentum' # 'AdaGrad' , 'RSMprop' , 'Momentum' , 'Adam'
    #  parameters for AdaGrad 
    grad = gradL(data, p)
    n = gradL(data, p)**2
    ep = 1e-8
    #  parameters for RSMprop
    alpha = 0.9
    sigma = np.sqrt((1-alpha) * grad**2 + ep)
    #  paraameters for Momentum
    v = np.array([0, 0])
    lamda = 0.9
    #  paraameters for Adam
    beta1 = np.array([0.9, 0.9])
    beta2 = np.array([0.999, 0.999])
    m = beta1 * np.array([0.0, 0.0]) + (1 - beta1) * grad
    v_adam = beta2 * np.array([0.0, 0.0]) + (1 - beta2) * grad**2


    criteria_stop = []
    i = 1
    converged = np.sqrt(sum( (grad)**2)) < tol
    #lamda * v - l * grad)**2
    while ~converged and (i < 1000):
        # print('try ' + str(i))
        converged = np.sqrt(sum((grad) ** 2)) < tol
        grad = gradL(data, p)
        criteria_stop += [np.sqrt(sum( (grad)**2))]
    
        if Method == 'AdaGrad':
            n += grad**2
            l_t = l/np.sqrt(n + ep)
            p = p - l_t * grad  # update fitting parameters
        elif Method == 'RSMprop':
            sigma = np.sqrt(alpha * sigma**2 + (1-alpha) * grad**2 + ep)
            p = p - l/sigma * grad
        elif Method == 'Momentum':
            v = lamda * v - l * grad
            p = p + v
        else: #default is Adam
            m_hat = m / (1 - beta1)
            v_adam_hat = v_adam / (1 - beta2)
            p = p - l * m_hat / (np.sqrt(v_adam_hat) + ep)
            m = beta1 * m + (1 - beta1) * grad
            v_adam = beta2 * v_adam + (1 - beta2) * grad**2    
        i += 1
    return np.array(p).astype(int), converged, criteria_stop

def plotresult(data_ori, p):
    BM_initial_fit = np.mean(data_ori[:p[0]])
    BM_final_fit = np.mean(data_ori[p[1]:])
    
    curve_initial_fit = BM_initial_fit * np.ones(p[0])
    t_drop_fit = p[1] - p[0]
    velocity = (BM_final_fit-BM_initial_fit)/t_drop_fit
    curve_drop_fit = np.linspace(BM_initial_fit, BM_final_fit, t_drop_fit)
    curve_final_fit = BM_final_fit * np.ones(len(data_ori) - p[1])
    data_fit = np.append(np.append(curve_initial_fit,curve_drop_fit), curve_final_fit)
    fig, ax = plt.subplots(figsize=(10,8))
    ax.plot(data_ori)
    t_fit = np.arange(len(data_fit))
    ax.plot(data_fit)
    ax.set_xlabel('Frame')
    ax.set_ylabel('BM (nm)')
    plt.show()
    return BM_initial_fit, BM_final_fit, velocity



if __name__ == "__main__":
    ##  parameters
    t_initial = 200
    t_final = 500

    ##  simluate data
    t, data_ori = gen_assembly(BM=[BM_initial, BM_final], t_change=[t_initial, t_final])
    data = nordata(data_ori)
    p_soln = [t_initial, t_final]

    ##  gradient descent
    p_initial = np.array([250, 300])
    p, converged, criteria_stop = gradescent(data, p_initial, tol=5e-2)
    BM_initial_fit, BM_final_fit, velocity = plotresult(data_ori, p)

    plt.figure()
    plt.plot(criteria_stop)

