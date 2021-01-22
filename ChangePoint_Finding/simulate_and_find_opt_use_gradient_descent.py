# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 13:29:55 2020

find two point to mininize square error of
BM before(const.) + one dropping line + BM after(const.)
loss function is un-differentiable
"""


##
import numpy as np
import matplotlib.pyplot as plt


## papameters
p = [150,30,350,15]

##
def slopecurve(p):
    t = np.array(0)
    t = np.append(t, np.array(p).astype('int64'))
    curve = np.empty(0)
    # for i in range(len(p)-1):
    t1 = int(p[0])
    t2 = int(p[1])
    c1 = np.nanmean(data[0:t1])
    c2 = np.nanmean(data[t2:])
        # curve = np.append(curve, np.linspace(np.mean(data[t[i]:t[i+1]]), 
        #                                       np.mean(data[t[i+1]:t[i+2]]), 
        #                                       abs(t[i+2] - t[i+1])))
    return np.linspace(c1, c2, abs(t2 - t1))
    # return curve

def lossfun(p):
    # p: [t1, t2]
    t1 = int(p[0])
    t2 = int(p[1])
    c1 = np.mean(data[0:t1])
    c2 = np.mean(data[t2:])
    sse1 = sum((data[0:t1] - c1)**2)
    sse2 = sum((data[t2:] - c2)**2)
    sse3 = sum((data[t1:t2] - slopecurve(p))**2)
    return sse1 + sse2 + sse3

def nordata(data):
    m = np.mean(data)
    s = np.std(data)
    return (data-m)/s

def gradL(p):
    grad_t1 = lossfun([p[0]+1, p[1]]) - lossfun(p)
    grad_t2 = lossfun([p[0], p[1]+1]) - lossfun(p)
    return np.array([grad_t1, grad_t2])

def gradescent(p_initial):
    p = p_initial
    l = 2 # learning rate
    Method = 'Adam' # 'AdaGrad' , 'RSMprop' , 'Momentum' , 'Adam'
    #  parameters for AdaGrad 
    l_t = np.array([1, 1])
    grad = gradL(p)
    n = gradL(p)**2
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
    #lamda * v - l * grad)**2
    while (np.sqrt(sum( (grad)**2)) > 1e-2) and (i < 1000): 
        # print('try ' + str(i))        
        grad = gradL(p)
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
    return np.array(p).astype(int)

def plotresult(p):
    BM_initial_fit = np.mean(data_ori[:p[0]])
    BM_final_fit = np.mean(data_ori[p[1]:])
    
    curve_initial_fit = BM_initial_fit * np.ones(p[0])
    t_drop_fit = p[1] - p[0]
    curve_drop_fit = np.linspace(BM_initial_fit, BM_final_fit, t_drop_fit)
    curve_final_fit = BM_final_fit * np.ones(len(data) - p[1])
    data_fit = np.append(np.append(curve_initial_fit,curve_drop_fit), curve_final_fit)
    plt.figure()
    plt.plot(t, data_ori)
    t_fit = np.arange(len(data_fit))
    plt.plot(t_fit, data_fit)



## simluate data
#  setting parameters
BM_initial = 35
sd_initial = 5
t_initial = 150
BM_final = 10
sd_final = 3.5
t_final = 300
velocity = 0.1
t_drop = int(abs((BM_final - BM_initial)/velocity))

curve_initial = BM_initial * np.ones(t_initial) + np.random.normal(0, sd_initial, t_initial) 
curve_drop = np.linspace(BM_initial, BM_final, t_drop) + np.random.normal(0, (sd_initial+sd_final)/2, t_drop) 
# curve_drop = slopecurve([t_initial, t_initial+t_drop]) + np.random.normal(0, (sd_initial+sd_final)/2, t_drop) 
curve_final = BM_final * np.ones(t_final) + np.random.normal(0, sd_final, t_final) 

data_ori = np.append(np.append(curve_initial,curve_drop), curve_final)
t = np.linspace(1, t_initial+t_final+t_drop, t_initial+t_final+t_drop)
data = nordata(data_ori)

p_ideal_soln = [t_initial, t_initial+t_drop]

## gradient descent 

p_initial = np.array([250, 300])

p = gradescent(p_initial)
p = p_initial
l = 3 # learning rate


Method = 'Momentum' # 'AdaGrad' , 'RSMprop' , 'Momentum' , 'Adam'
#  parameters for AdaGrad
l_t = np.array([1, 1])
grad = gradL(p)
n = gradL(p)**2
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
#lamda * v - l * grad)**2
while (np.sqrt(sum( (grad)**2)) > 1e-5) and (i < 500):
    print('try ' + str(i))
    
    grad = gradL(p)
    criteria_stop += [np.sqrt(sum( (grad)**2))]
    # n += grad**2
    # l_t = l/np.sqrt(n + ep)
    # sigma = np.sqrt(alpha * sigma**2 + (1-alpha) * grad**2 + ep)
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
p = np.array(p).astype(int)
plt.figure()

plt.plot(criteria_stop)



##  plot result
plotresult(p)


# BM_initial_fit = np.mean(data_ori[:p[0]])
# BM_final_fit = np.mean(data_ori[p[1]:])

# curve_initial_fit = BM_initial_fit * np.ones(p[0])
# t_drop_fit = p[1] - p[0]
# curve_drop_fit = np.linspace(BM_initial_fit, BM_final_fit, t_drop_fit)
# curve_final_fit = BM_final_fit * np.ones(len(data) - p[1])
# data_fit = np.append(np.append(curve_initial_fit,curve_drop_fit), curve_final_fit)
# plt.figure()
# plt.plot(t, data_ori)
# t_fit = np.arange(len(data_fit))
# plt.plot(t_fit, data_fit)