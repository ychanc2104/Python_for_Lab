# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 09:43:19 2020

calculate xy-ration and xy BM for given xy position csv

@author: YCC
"""
###  input parameters
path_folder = r'C:\Users\OT-hwLi-lab\Desktop\YCC\20210127\qdot655\3281bp\3\4-200ms-110uM_BME'

window_std = 20
window_avg = 1
factor_p2n = 10000/180 # nm/pixel
mode_select_xyratio = 'on'

upper_r_xy = 1.2
lower_r_xy = 0.8

###  import used module
import numpy as np
import csv
from glob import glob
import os
import datetime
import matplotlib.pyplot as plt


### getting parameters
today = datetime.datetime.now()
filename_time = str(today.year)+str(today.month)+str(today.day)

### read xy position file
file_path = glob(path_folder+'/*xy and sigma xy.csv')[-1]
# cwd = os.getcwd() + '\\' + file_name[0]
with open(file_path, newline='') as csvfile:
    rows = csv.reader(csvfile)
    # bead_number = len([x for x in rows])
    x = []
    y = []
    sx = []
    sy = []
    intensity = []
    sheet = []
    for row in rows:
        sheet += [row]
    bead_number = int(len(sheet[0])/5)
    frame_number = len(sheet) - 1
    bead_namexy = ['BMx '+str(i+1) for i in range(bead_number)]+['BMy '+str(i+1) for i in range(bead_number)]+['bsize '+str(i+1) for i in range(bead_number)]+['I '+str(i+1) for i in range(bead_number)]
    for i in range(frame_number):
        x += [sheet[i+1][0:bead_number]] 
        y += [sheet[i+1][(bead_number):bead_number*2]]
        sx += [sheet[i+1][(bead_number*2):bead_number*3]]
        sy += [sheet[i+1][(bead_number*3):bead_number*4]]
        intensity += [sheet[i+1][(bead_number*4):bead_number*5]]
        
    x = np.array(x, dtype = np.float32)
    y = np.array(y, dtype = np.float32)
    sx = np.array(sx, dtype = np.float32)
    sy = np.array(sy, dtype = np.float32)
    intensity = np.array(intensity, dtype = np.float32)
    BMx = []
    BMy = []
    for k in range(bead_number):
        print('start analyzing bead' + str(k+1))
        for i in range(frame_number-window_std+1): #silding window
            x_slice = x[i:window_std+i,k] # choose 'window_std' point at a time
            y_slice = y[i:window_std+i,k]
            for j in range(window_std): #for ignoring 0 element
                x_slice_del = []
                y_slice_del = []
                xy_i_del = []
                if (x_slice[j] <= 0) or (y_slice[j] <= 0):
                    xy_i_del += [j]
            if xy_i_del != []:  # there are fitting error points   
                x_slice_del = np.delete(x_slice,xy_i_del)
                y_slice_del = np.delete(y_slice,xy_i_del)               
                BMx += [np.std(x_slice_del,ddof = 1)]
                BMy += [np.std(y_slice_del,ddof = 1)]
            else:
                BMx += [np.std(x_slice,ddof = 1)]
                BMy += [np.std(y_slice,ddof = 1)]
    BMx = factor_p2n * np.reshape(BMx, (bead_number,frame_number-window_std+1)).transpose()
    BMy = factor_p2n * np.reshape(BMy, (bead_number,frame_number-window_std+1)).transpose()
    BMz = sx*sy
    I = intensity

###  select xy ratio
BMx_mean = np.mean(np.transpose(BMx),1)
if mode_select_xyratio == 'on': #select xy ratio

    # ratio_xy = (sum(BMx)/sum(BMy))
    ratio_xy = np.mean(BMx/BMy, axis=0)
    i_rxy_dele = [x for x in range(bead_number) if (ratio_xy[x] > upper_r_xy) or (ratio_xy[x] < lower_r_xy) or (BMx_mean[x] > 200)]
    BMx = np.delete(BMx,i_rxy_dele,axis = 1) # delete  (bead#)
    BMy = np.delete(BMy,i_rxy_dele,axis = 1)
    BMz = np.delete(BMz,i_rxy_dele,axis = 1)
    BMI = np.delete(I,i_rxy_dele,axis = 1)
    bead_namexy = np.delete(bead_namexy,i_rxy_dele+list(np.array(i_rxy_dele)+bead_number)+list(np.array(i_rxy_dele)+2*bead_number)+list(np.array(i_rxy_dele)+3*bead_number))
    bead_sele_number = np.shape(BMx)[1]
    BMxyz = np.empty((frame_number, bead_sele_number*4))
    BMxyz[:frame_number-window_std+1,:bead_sele_number] = BMx
    BMxyz[:frame_number-window_std+1,bead_sele_number:bead_sele_number*2] = BMy
    BMxyz[:,bead_sele_number*2:bead_sele_number*3] = BMz
    BMxyz[:,bead_sele_number*3:] = BMI
    # bead_namexy
else:

    BMxyz = np.empty((frame_number, bead_number*3))
    BMxyz[:frame_number-window_std+1,:bead_sele_number] = BMx
    BMxyz[:frame_number-window_std+1,bead_sele_number:bead_sele_number*2] = BMy
    BMxyz[:,bead_sele_number*2:] = BMz
    
BMx_mean = np.mean(np.transpose(BMx),1)

###  save to csv
with open(path_folder +'/' +filename_time + 'selectxy-' + mode_select_xyratio + '-xyBM.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(bead_namexy)
    writer.writerows(BMxyz)

plt.figure
plt.hist(BMx_mean,20)
print('finish saving')







