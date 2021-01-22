# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 12:53:07 2020

@author: YCC

custom-made gradient descent with four optimization methods 
('AdaGrad' , 'RSMprop' , 'Momentum' , 'Adam')
"""


########################################
## input papameters
path = 'C:/Users/hwlab/Desktop/YCC/YCH/test'
avg_window = 20 # smooth fix window size
fps = 33/avg_window # Hz
criteria_BM = 58 # center BM of qualified tether
s = 10 # criteria for range of good tether
#######################################



##  import used module
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from os import listdir
import csv

##  import needed function
##  read data, #beads, #frames from csv
def load_data(csvfile):
    rows = csv.reader(csvfile)
    sheet = []
    for row in rows:
        sheet += [row]
    bead_number = int(len(sheet[0]) - 1)
    frame_number = len(sheet)
    return sheet, bead_number, frame_number

##  fix-window average, default window size = 20, append nan if #nan in window > 50%
def avg_fixwin(data, windowsize = 20):   
    n_frame = np.shape(data)[0]    
    n_data = np.shape(data)[1]
    n = np.ceil(n_frame/windowsize).astype('int')
    data_filter = np.empty(0)
    for i in range(n_data):  # deal with multi-sets of DNA tethers
        data_input = data[:,i]
        for i in range(n): # # of loop for fix window avg.
            if sum(np.isnan(data_input[(i)*windowsize: min( (i+1)*windowsize, n_frame-1)])) < windowsize/2:
                data_filter = np.append(data_filter, 
                                        np.nanmean( data_input[ i*windowsize: min( (i+1)*windowsize, n_frame-1)]))
            else: 
                data_filter = np.append(data_filter, np.nan)     
    data_filter = np.reshape(data_filter,(n_data,n))    
    return data_filter.T

##  remove nan from data and repalce it with previous data
def removenan(data, low = 0.1, high = 80):
    n = len(data)
    for i in range(n):
        n_rand = np.ceil(np.random.random(1) * 20).astype('int')
        # n_rand = 20
        if i > n_rand:
            if (data[i] > low) and (data[i] < high):
                data[i] = data[i]
            else:
                data[i] = data[i - n_rand]
    return data

##  data normalization
def nordata(data):
    m = np.nanmean(data)
    s = np.nanstd(data)
    return (data-m)/s

##  get two mean values, change points and dropping curve
def slopecurve(p, data): # p: [t1, t2]
    t1 = int(p[0])
    t2 = int(p[1])
    c1 = np.nanmean(data[0:t1]) # initial BM
    c2 = np.nanmean(data[t2:])  # final BM
    return t1, t2, c1, c2, np.linspace(c1, c2, abs(t2 - t1))

##  define loss function by sum of square error
def lossfun(p): # p: [t1, t2]
    t1, t2, c1, c2, dropcurve = slopecurve(p, data)
    sse1 = np.nansum((data[0:t1] - c1)**2) # initial BM
    sse2 = np.nansum((data[t2:] - c2)**2)  # final BM
    sse3 = np.nansum((data[t1:t2] - dropcurve)**2) # velocity
    return sse1 + sse2 + sse3

##  calculate discrete difference 
def gradL(p):
    grad_t1 = lossfun([p[0]+1, p[1]]) - lossfun(p)
    grad_t2 = lossfun([p[0], p[1]+1]) - lossfun(p)
    return np.array([grad_t1, grad_t2])

##  gradient descent with four methods, default is Adam
def gradescent(p_initial, Method = 'Adam'): # method can be ignore, default = Adam
    p = p_initial
    l = 0.18 # learning rate, if fitting unstable, change this   
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

##  plot and save figure and get velocity
def plotresult(p):
    t1, t2, BM_initial_fit, BM_final_fit, curve_drop_fit = slopecurve(p, data_ori)
    
    curve_initial_fit = BM_initial_fit * np.ones(t1)
    t_drop_fit = t2 - t1
    curve_final_fit = BM_final_fit * np.ones(len(data) - t2)
    data_fit = np.concatenate((curve_initial_fit, curve_drop_fit, curve_final_fit))
    velocity = (BM_final_fit - BM_initial_fit)/t_drop_fit * fps*60 # v (nm/min)
    plt.figure()
    t = np.arange(len(data_ori))/fps
    t_fit = np.arange(len(data_fit))/fps
    plt.plot(t, data_ori,'.')
    plt.plot(t_fit, data_fit,'r-')
    plt.xlabel('Time (s)')
    plt.ylabel('BM (nm)')
    # plt.savefig('conc-' + label +'-'+ str(fig_label) + '.png')
    return velocity


###  read all data folders
csv_path = []
folders = listdir(path)
p_label = []

###  get all csv path in all folders
for folder in folders:
    csv_path += [x for x in glob(path +'/' + folder + '/*.csv')]
    p_label += [folder] * len(glob(path + '/' + folder + '/*.csv'))
    
###  initial values for storage
p_fit_label = []
p_fit = []
velocity_fit = []
fig_label = 1

###  calculate all velocity of each csv files
for file, label in zip(csv_path, p_label): # for loop for each .csv file
    print('caculating file in path: ' + file)    
    with open(file, newline='') as csvfile:
        sheet, bead_number, frame_number = load_data(csvfile)
        data = np.array(sheet, dtype = np.float32)
        time = data[:,0]
        BM = data[0:13000,1:]
        BM_initial = np.nanmean(data[time < 0 ,1:].T, 1)
        BM_after = np.nanmean(data[(time >= 0) & (time <= 20) ,1:].T, 1)
        BM_select = BM[:,(BM_initial > criteria_BM - s) & (BM_initial < criteria_BM + s) & (abs(BM_initial - BM_after) < s*2)]
        BM_select_filter = avg_fixwin(BM_select[:])
        for i in range(np.shape(BM_select_filter)[1]):
            
            data_ori = np.copy(BM_select_filter[:,i])
            data_remove = removenan(np.copy(data_ori), 0.1, 80)
            data = nordata(data_remove)
            p_initial = np.array([50, 70])
            try:
                p = gradescent(p_initial, Method = 'Momentum')
                v = plotresult(p)
                p_fit += [p]
                p_fit_label += [label]
                velocity_fit += [v]
                fig_label += 1
            except ValueError: # skip error
                p = np.empty(2)
                
