# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 09:43:19 2020

calculate xy-ration and xy BM for given xy position csv

@author: YCC
"""
###  input parameters
path_folder = r'F:\YCC\20210205\4-1895bp\1-200ms-440uM_BME_gain20'

window_std = 20
window_avg = 1
factor_p2n = 10000/180 # nm/pixel
mode_select_xyratio = 'on'
avg_fps = 4.99
upper_r_xy = 1.2
lower_r_xy = 0.8

###  import used module
import numpy as np
import csv
from glob import glob
import os
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import math

### input 1D array data, output: (row, column) = (frame, bead)
def get_attrs(data_col, bead_number, frame_acquired):
    data_col = np.array(data_col)
    data_col_reshape = np.reshape(data_col, (frame_acquired, bead_number))
    return data_col_reshape

### get name and bead number to be saved
def get_name(name, bead_number):
    name = [f'{name}_{i+1}' for i in range(bead_number)]
    return np.array(name)

### getting date
def get_date():
    today = datetime.datetime.now()
    filename_time = str(today.year)+str(today.month)+str(today.day) # y/m/d
    return filename_time

# ### data:1D array for a bead
# def calBM_2D(data, window = 20, method = 'silding'):

#   return 

### data:1D array for a bead, BM: 
def calBM(data, window = 20, method = 'silding'):
  if method == 'silding': # overlapping
    iteration = len(data) - window + 1 # silding window
    BM_s = []
    for i in range(iteration):
      data_pre = data[i: i+window]
      BM_s += [np.std(data_pre[data_pre > 0], ddof = 1)]
      BM = BM_s  
  else: # fix, non-overlapping
    iteration = int(len(data)/window)  # fix window
    BM_f = []
    for i in range(iteration):
      data_pre = data[i*window: (i+1)*window]
      BM_f += [np.std(data_pre[data_pre > 0], ddof = 1)]
      BM = BM_f
  return BM




##  
def gather_sheets(writer, element, bead_number, frame_acquired, df_time):
    name = get_name(element, bead_number)
    data = get_attrs(df[element], bead_number, frame_acquired)
    df_reshape = pd.DataFrame(data=data, columns=name)
    df_reshape.insert(0, 'time', df_time)
    df_reshape.to_excel(writer, sheet_name=element)
    return df_reshape


### get sx*sy and sx/sy ,and write reshape data to excel
def save_reshape_data(df, path_folder, filename_time):
    bead_number = int(max(df['aoi']))
    frame_acquired = int(len(df['x'])/bead_number)
    df.insert(5, 'sx_sy', df['sx']*df['sy'])
    df.insert(6, 'sx_over_sy', df['sx']/df['sy'])
    df_time = pd.DataFrame(data=np.arange(0,frame_acquired)/5)
    writer = pd.ExcelWriter(os.path.join(path_folder, f'{filename_time}-fitresults_reshape.xlsx'))
    for i, element in enumerate(df.columns):
        if i > 1:
            if element == 'x':
                df_reshape = gather_sheets(writer, element, bead_number, frame_acquired, df_time)
                x_2D = np.array(df_reshape)[:,1:]
                
            elif element == 'y':
                df_reshape = gather_sheets(writer, element, bead_number, frame_acquired, df_time)
                y_2D = np.array(df_reshape)[:,1:]
            
            elif element == 'sx':
                df_reshape = gather_sheets(writer, element, bead_number, frame_acquired, df_time)
                sx_2D = np.array(df_reshape)[:,1:]
            
            elif element == 'sy':
                df_reshape = gather_sheets(writer, element, bead_number, frame_acquired, df_time)
                sy_2D = np.array(df_reshape)[:,1:]
                
            else:
                df_reshape = gather_sheets(writer, element, bead_number, frame_acquired, df_time)
    writer.save()
    return x_2D, y_2D, sx_2D, sy_2D


### getting date
filename_time = get_date()

### read tracking result file
file_path = glob(os.path.join(path_folder, '*-fitresults.csv'))[-1]
df = pd.read_csv(file_path)
x_2D, y_2D, sx_2D, sy_2D = save_reshape_data(df, path_folder, filename_time)


BMx_silding = []
BMx_fixing = []
BMy_silding = []
BMy_fixing = []
BM_all = []
for (x_1D, y_1D) in zip(x_2D.T, y_2D.T):
    BMx_silding += [calBM(x_1D, window = 20, method = 'silding')]
    BMx_fixing += [calBM(x_1D, window = 20, method = 'fixing')]
    BMy_silding += [calBM(y_1D, window = 20, method = 'silding')]
    BMy_fixing += [calBM(y_1D, window = 20, method = 'fixing')]
BM_all += [[BMx_silding]] + [[BMx_fixing]] + [[BMy_silding]] + [[BMy_fixing]]
    
    
BMx_silding = factor_p2n * np.array(BMx_silding).T
BMy_silding = factor_p2n * np.array(BMy_silding).T
xy_ratio_silding = np.mean(BMx_silding/BMy_silding, 0)
BMx_fixing = factor_p2n * np.array(BMx_fixing).T
BMy_fixing = factor_p2n * np.array(BMy_fixing).T
xy_ratio_fixing = np.mean(BMx_fixing/BMy_fixing, 0)
sxsy_2_ratio = np.mean(sx_2D/sy_2D, 0)**2

ratio = np.array([xy_ratio_silding] + [xy_ratio_fixing] + [sxsy_2_ratio]).T
ratio = np.nan_to_num(ratio)

c = (ratio>0.8) & (ratio <1.2)
criteria = []
for row_boolean in c:
   criteria += [all(row_boolean)]
criteria = np.array(criteria)
    


BMxy = [BMx_silding] + [BMy_silding] + [BMx_fixing] + [BMy_fixing] + [ratio]


sheet_names = ['BMx_silding', 'BMy_silding', 'BMx_fixing', 'BMy_fixing', 'ratio_test']
writer = pd.ExcelWriter(os.path.join(path_folder, f'{filename_time}-fitresults_reshape_analyze.xlsx'))
frames_acquired = x_2D.shape[0]
dt = (x_2D.shape[0] - BMx_silding.shape[0] + 1)/avg_fps/2
for BM, sheet_name in zip(BMxy, sheet_names):
    if sheet_name != 'ratio_test':
        beads = get_name(sheet_name, BM.shape[1])
        df_reshape_analyze = pd.DataFrame(data=BM[:,criteria], columns=beads[criteria])
        df_reshape_analyze.insert(0, 'time', dt + np.arange(0, BM.shape[0])/avg_fps*math.floor(frames_acquired/BM.shape[0]))
        df_reshape_analyze.to_excel(writer, sheet_name=sheet_name)
writer.save()

# bead_number = int(max(df['aoi']))
# frame_acquired = int(len(df['x'])/bead_number)
# df.insert(5, 'sx_sy', df['sx']*df['sy'])
# df.insert(6, 'sx_over_sy', df['sx']/df['sy'])


# writer = pd.ExcelWriter(os.path.join(path_folder, f'{filename_time}-fitresults_reshape.xlsx'))
# for i, element in enumerate(df.columns):
#     if i > -1:
#         name = get_name(element, bead_number)
#         data = get_attrs(df[element], bead_number, frame_acquired)
#         df_reshape = pd.DataFrame(data=data, columns=name)
#         df_reshape.to_excel(writer, sheet_name=element)
# writer.save()





# with open(file_path, newline='') as csvfile:
#     rows = csv.reader(csvfile)
#     # bead_number = len([x for x in rows])
#     x = []
#     y = []
#     sx = []
#     sy = []
#     intensity = []
#     sheet = []
#     for row in rows:
#         sheet += [row]
#     bead_number = int(len(sheet[0])/5)
#     frame_number = len(sheet) - 1
#     bead_namexy = ['BMx '+str(i+1) for i in range(bead_number)]+['BMy '+str(i+1) for i in range(bead_number)]+['bsize '+str(i+1) for i in range(bead_number)]+['I '+str(i+1) for i in range(bead_number)]
#     for i in range(frame_number):
#         x += [sheet[i+1][0:bead_number]]
#         y += [sheet[i+1][(bead_number):bead_number*2]]
#         sx += [sheet[i+1][(bead_number*2):bead_number*3]]
#         sy += [sheet[i+1][(bead_number*3):bead_number*4]]
#         intensity += [sheet[i+1][(bead_number*4):bead_number*5]]
#
#     x = np.array(x, dtype = np.float32)
#     y = np.array(y, dtype = np.float32)
#     sx = np.array(sx, dtype = np.float32)
#     sy = np.array(sy, dtype = np.float32)
#     intensity = np.array(intensity, dtype = np.float32)
#     BMx = []
#     BMy = []
#     for k in range(bead_number):
#         print('start analyzing bead' + str(k+1))
#         for i in range(frame_number-window_std+1): #silding window
#             x_slice = x[i:window_std+i,k] # choose 'window_std' point at a time
#             y_slice = y[i:window_std+i,k]
#             for j in range(window_std): #for ignoring 0 element
#                 x_slice_del = []
#                 y_slice_del = []
#                 xy_i_del = []
#                 if (x_slice[j] <= 0) or (y_slice[j] <= 0):
#                     xy_i_del += [j]
#             if xy_i_del != []:  # there are fitting error points
#                 x_slice_del = np.delete(x_slice,xy_i_del)
#                 y_slice_del = np.delete(y_slice,xy_i_del)
#                 BMx += [np.std(x_slice_del,ddof = 1)]
#                 BMy += [np.std(y_slice_del,ddof = 1)]
#             else:
#                 BMx += [np.std(x_slice,ddof = 1)]
#                 BMy += [np.std(y_slice,ddof = 1)]
#     BMx = factor_p2n * np.reshape(BMx, (bead_number,frame_number-window_std+1)).transpose()
#     BMy = factor_p2n * np.reshape(BMy, (bead_number,frame_number-window_std+1)).transpose()
#     BMz = sx*sy
#     I = intensity
#
# ###  select xy ratio
# BMx_mean = np.mean(np.transpose(BMx),1)
# if mode_select_xyratio == 'on': #select xy ratio
#
#     # ratio_xy = (sum(BMx)/sum(BMy))
#     ratio_xy = np.mean(BMx/BMy, axis=0)
#     i_rxy_dele = [x for x in range(bead_number) if (ratio_xy[x] > upper_r_xy) or (ratio_xy[x] < lower_r_xy) or (BMx_mean[x] > 200)]
#     BMx = np.delete(BMx,i_rxy_dele,axis = 1) # delete  (bead#)
#     BMy = np.delete(BMy,i_rxy_dele,axis = 1)
#     BMz = np.delete(BMz,i_rxy_dele,axis = 1)
#     BMI = np.delete(I,i_rxy_dele,axis = 1)
#     bead_namexy = np.delete(bead_namexy,i_rxy_dele+list(np.array(i_rxy_dele)+bead_number)+list(np.array(i_rxy_dele)+2*bead_number)+list(np.array(i_rxy_dele)+3*bead_number))
#     bead_sele_number = np.shape(BMx)[1]
#     BMxyz = np.empty((frame_number, bead_sele_number*4))
#     BMxyz[:frame_number-window_std+1,:bead_sele_number] = BMx
#     BMxyz[:frame_number-window_std+1,bead_sele_number:bead_sele_number*2] = BMy
#     BMxyz[:,bead_sele_number*2:bead_sele_number*3] = BMz
#     BMxyz[:,bead_sele_number*3:] = BMI
#     # bead_namexy
# else:
#
#     BMxyz = np.empty((frame_number, bead_number*3))
#     BMxyz[:frame_number-window_std+1,:bead_sele_number] = BMx
#     BMxyz[:frame_number-window_std+1,bead_sele_number:bead_sele_number*2] = BMy
#     BMxyz[:,bead_sele_number*2:] = BMz
#
# BMx_mean = np.mean(np.transpose(BMx),1)
#
# ###  save to csv
# with open(path_folder +'/' +filename_time + 'selectxy-' + mode_select_xyratio + '-xyBM.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(bead_namexy)
#     writer.writerows(BMxyz)
#
# plt.figure
# plt.hist(BMx_mean,20)
# print('finish saving')







