# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 09:43:19 2020

calculate xy-ration and xy BM for given xy position csv

@author: YCC
"""
###  input parameters
path_folder = r'C:\Users\pine\Desktop\1-200ms-440uM_BME_gain20'

window = 20
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
from sklearn.decomposition import PCA


### input 1D array data, output: (row, column) = (frame, bead)
def get_attrs(data_col, bead_number, frame_acquired):
    data_col = np.array(data_col)
    data_col_reshape = np.reshape(data_col, (frame_acquired, bead_number))
    return data_col_reshape

### get name and bead number to be saved, 1st col is time
def get_columns(name, bead_number):
    columns = ['time'] + [f'{name}_{i}' for i in range(bead_number)]
    return np.array(columns)

### getting date
def get_date():
    filename_time = datetime.datetime.today().strftime('%Y-%m-%d') # yy-mm-dd
    return filename_time

### get analyzed sheet names
def get_analyzed_sheet_names():
    return ['BMx_sliding', 'BMy_sliding', 'BMx_fixing', 'BMy_fixing', 
            'sx_sy', 'xy_ratio_sliding', 'xy_ratio_fixing', 'sx_over_sy_squared',
            'avg_attrs', 'std_attrs']
  
### get analyzed sheet names
def get_reshape_sheet_names():
    return ['amplitude', 'sx', 'sy', 'x', 'y', 'theta_deg', 'offset', 'intensity', 'intensity_integral', 'ss_res']


### data:1D numpy array for a bead, BM: 1D numpy array
def calBM_1D(data, window = 20, factor_p2n = 10000/180, method = 'sliding'):
  if method == 'sliding': # overlapping
    iteration = len(data) - window + 1 # silding window
    BM_s = []
    for i in range(iteration):
      data_pre = data[i: i+window]
      BM_s += [factor_p2n * np.std(data_pre[data_pre > 0], ddof = 1)]
    BM = BM_s  
  else: # fix, non-overlapping
    iteration = int(len(data)/window)  # fix window
    BM_f = []
    for i in range(iteration):
      data_pre = data[i*window: (i+1)*window]
      BM_f += [factor_p2n * np.std(data_pre[data_pre > 0], ddof = 1)]
    BM = BM_f
  return np.array(BM)

def calBM_2D(data_2D, avg_fps, window = 20, factor_p2n = 10000/180):
    ##  get BM of each beads
    BM_sliding = []
    BM_fixing = []
    for data_1D in data_2D.T:
        BM_sliding += [calBM_1D(data_1D, window = window, method = 'sliding')]
        BM_fixing += [calBM_1D(data_1D, window = window, method = 'fixing')]
    BM_sliding = np.array(BM_sliding).T
    BM_fixing = np.array(BM_fixing).T

    return BM_sliding, BM_fixing
        
##  cal BMx and BMy ratio
def get_xy_ratio(*args):
    xy_ratio = []
    for data in args:
        xy_ratio += [data[0]/data[1]]
    return xy_ratio

##  save each attributes to each sheets, data:2D array
def gather_reshape_sheets(df, sheet_name, bead_number, frame_acquired, dt, avg_fps):
    name = get_columns(sheet_name, bead_number)
    data = get_attrs(df[sheet_name], bead_number, frame_acquired)
    data = np.array(append_time([data], avg_fps, frame_acquired))
    data = np.reshape(data, (frame_acquired, bead_number+1))             
    df_reshape = pd.DataFrame(data=data, columns=name)
    return df_reshape

### get reshape data all
def get_reshape_data(df, avg_fps, window = 20):
    bead_number = int(max(df['aoi']))
    frame_acquired = int(len(df['x'])/bead_number)
    df_reshape = dict()
    dt = window/2/avg_fps
    for i, sheet_name in enumerate(df.columns):
        if i > 1:
            df_reshape[sheet_name] = gather_reshape_sheets(df, sheet_name, bead_number, frame_acquired, dt, avg_fps)
    return df_reshape

##  add time axis into first column, data: list of 2D array,(r,c)=(frame,bead)
def append_time(analyzed_data, avg_fps, frames_acquired, window=20):
    dt = window/2/avg_fps
    analyzed_append_data = []
    for data in analyzed_data:
        time = dt + np.arange(0, data.shape[0])/avg_fps*math.floor(frames_acquired/data.shape[0])
        time = np.reshape(time, (-1,1))
        analyzed_append_data += [np.append(time, data, axis=1)]
        
    return analyzed_append_data

# get anaylyzed data
def get_analyzed_data(df_reshape, window, avg_fps, factor_p2n=factor_p2n):
    x_2D = np.array(df_reshape['x'])[:,1:]
    y_2D = np.array(df_reshape['y'])[:,1:]
    sx_2D = np.array(df_reshape['sx'])[:,1:]
    sy_2D = np.array(df_reshape['sy'])[:,1:]    
    bead_number = x_2D.shape[1]
    frame_acquired = x_2D.shape[0]
    
    BMx_sliding, BMx_fixing = calBM_2D(x_2D, avg_fps, factor_p2n=factor_p2n)
    BMy_sliding, BMy_fixing = calBM_2D(y_2D, avg_fps, factor_p2n=factor_p2n)
    sx_sy = sx_2D * sy_2D
    xy_ratio = get_xy_ratio([BMx_sliding, BMy_sliding], [BMx_fixing, BMy_fixing], [sx_2D**2, sy_2D**2])
    data_analyzed_avg, data_analyzed_std = avg_std_operator(BMx_sliding, BMx_fixing, BMy_sliding, BMy_fixing, sx_sy, xy_ratio[0], xy_ratio[1], xy_ratio[2])    
    data_reshaped_avg, data_reshaped_std = df_reshape_avg_std_operator(df_reshape)
    
    data_avg_2D = np.append(data_analyzed_avg, data_reshaped_avg, axis=1)
    data_std_2D = np.append(data_analyzed_std, data_reshaped_std, axis=1)
    
    analyzed_data = [BMx_sliding, BMy_sliding, BMx_fixing, BMy_fixing, sx_sy, xy_ratio[0], xy_ratio[1], xy_ratio[2]]
    analyzed_data = append_time(analyzed_data, avg_fps, frame_acquired, window=20)
    analyzed_data = analyzed_data + [data_avg_2D, data_std_2D]

    analyzed_sheet_names = get_analyzed_sheet_names()
    df_reshape_analyzed = df_reshape.copy()
    # writer = pd.ExcelWriter(os.path.join(path_folder, f'{filename_time}-fitresults_reshape_analyzed.xlsx'))

    dt = window/2/avg_fps
    for data, sheet_name in zip(analyzed_data, analyzed_sheet_names):
        if sheet_name == 'avg_attrs':
            df_reshape_analyzed[sheet_name] = pd.DataFrame(data=data, columns=get_analyzed_sheet_names()[:-2]+get_reshape_sheet_names())
        elif sheet_name == 'std_attrs':
            df_reshape_analyzed[sheet_name] = pd.DataFrame(data=data, columns=get_analyzed_sheet_names()[:-2]+get_reshape_sheet_names())
        else:
            df_reshape_analyzed[sheet_name] = pd.DataFrame(data=data, columns=get_columns(sheet_name, bead_number))

    return df_reshape_analyzed


##  data average operator for multiple columns(2D-array), output: (r,c)=(beads,attrs)
def avg_std_operator(*args):
    data_avg_2D = []
    data_std_2D = []
    for data_2D in args:
        data_avg = []
        data_std = []
        for data in data_2D.T:
            data_avg += [np.mean(data, axis=0)]
            data_std += [np.std(data, axis=0, ddof=1)]
        data_avg_2D += [np.array(data_avg)]
        data_std_2D += [np.array(data_std)]
    return np.nan_to_num(data_avg_2D).T, np.nan_to_num(data_std_2D).T

##  get avg and std for reshaped DataFrame
def df_reshape_avg_std_operator(df_reshape):
    data_avg = []
    data_std = []
    for i, sheet_name in enumerate(df.columns):
        if i >1:
            data = np.array(df_reshape[sheet_name])[:,1:]
            data_avg += [np.mean(data, axis=0)]
            data_std += [np.std(data, axis=0, ddof=1)]
    return np.array(data_avg).T, np.array(data_std).T        



##  get selection criteria
def get_criteria(df_reshape_analyzed):
    ratio = df_reshape_analyzed['avg_attrs'][['xy_ratio_sliding', 'xy_ratio_fixing', 'sx_over_sy_squared']]
    ratio = np.nan_to_num(ratio)
    c = ((ratio>0.8) & (ratio <1.2))
    criteria = []
    for row_boolean in c:
        criteria += [all(row_boolean)]
    return np.array(criteria)


##  save all dictionary of DataFrame to excel sheets
def save_all_dict_df_to_excel(dict_df, path_folder, filename='fitresults_reshape_analyzed.xlsx'):
    sheet_names = get_analyzed_sheet_names() + get_analyzed_sheet_names()
    writer = pd.ExcelWriter(os.path.join(path_folder, f'{filename_time}-{filename}'))
    for sheet_name in sheet_names:
        df_save = dict_df[sheet_name]
        df_save.to_excel(writer, sheet_name=sheet_name, index=False)
    writer.save()

##  save selected dictionary of DataFrame to excel sheets
def save_selected_dict_df_to_excel(dict_df, path_folder, filename='fitresults_reshape_analyzed_selected.xlsx'):
    criteria = get_criteria(dict_df)
    criteria = np.append(np.array(True),criteria)
    sheet_names = get_analyzed_sheet_names() + get_analyzed_sheet_names()
    writer = pd.ExcelWriter(os.path.join(path_folder, f'{filename_time}-{filename}'))
    for sheet_name in sheet_names:
        if sheet_name != 'avg_attrs' and sheet_name != 'std_attrs':
            df_save = df_reshape_analyzed[sheet_name]
            data = np.array(df_save)[:,criteria]
            df_save_selected = pd.DataFrame(data=data, columns=df_save.columns[criteria])
            df_save_selected.to_excel(writer, sheet_name=sheet_name, index=False)
        else: # for avg_attrs and std_attrs sheets
            df_save = df_reshape_analyzed[sheet_name]
            data = np.array(df_save).T[:,criteria[1:]]
            df_save_selected = pd.DataFrame(data=data.T, index=get_columns('bead', 12)[1:][criteria[1:]], columns=df_save.columns)
            df_save_selected.to_excel(writer, sheet_name=sheet_name, index=True)
    writer.save()

##  save removed dictionary of DataFrame to excel sheets
def save_removed_dict_df_to_excel(dict_df, path_folder, filename='fitresults_reshape_analyzed_removed.xlsx'):
    criteria = get_criteria(dict_df)
    criteria = ~np.append(np.array(False),criteria)
    sheet_names = get_analyzed_sheet_names() + get_analyzed_sheet_names()
    writer = pd.ExcelWriter(os.path.join(path_folder, f'{filename_time}-{filename}'))
    for sheet_name in sheet_names:
        if sheet_name != 'avg_attrs' and sheet_name != 'std_attrs':
            df_save = df_reshape_analyzed[sheet_name]
            data = np.array(df_save)[:,criteria]
            df_save_selected = pd.DataFrame(data=data, columns=df_save.columns[criteria])
            df_save_selected.to_excel(writer, sheet_name=sheet_name, index=False)
        else: # for avg_attrs and std_attrs sheets
            df_save = df_reshape_analyzed[sheet_name]
            data = np.array(df_save).T[:,criteria[1:]]
            df_save_selected = pd.DataFrame(data=data.T, index=get_columns('bead', 12)[1:][criteria[1:]], columns=df_save.columns)
            df_save_selected.to_excel(writer, sheet_name=sheet_name, index=True)
    writer.save()


### normalize to mean = 0, std = 1
def normalize_data(data):
    data_nor = []
    for datum in data.T:
        mean = np.mean(datum)
        std = np.std(datum, ddof=1)
        datum_nor = (datum-mean)/std
        data_nor += [datum_nor]
    return np.array(data_nor).T


### getting date
filename_time = get_date()

### read tracking result file
file_path = glob(os.path.join(path_folder, '*-fitresults.csv'))[-1]
df = pd.read_csv(file_path)

df_reshape = get_reshape_data(df, avg_fps, window=window)
df_reshape_analyzed = get_analyzed_data(df_reshape, window, avg_fps, factor_p2n)

save_all_dict_df_to_excel(df_reshape_analyzed, path_folder, filename='fitresults_reshape_analyzed.xlsx')
save_selected_dict_df_to_excel(df_reshape_analyzed, path_folder, filename='fitresults_reshape_analyzed_selected.xlsx')
save_removed_dict_df_to_excel(df_reshape_analyzed, path_folder, filename='fitresults_reshape_analyzed_removed.xlsx')





# ### PCA analysis
# a = df_reshape_analyzed['avg_attrs']
# b = df_reshape_analyzed['std_attrs']
# c = np.append(a, b, axis=1)
# d = normalize_data(c)
# ##   validate by built-in-function
# pca = PCA(n_components=5)
# result = pca.fit(d)
# e = result.transform(d)

# plt.plot(e[:,0], e[:,1],'o')



