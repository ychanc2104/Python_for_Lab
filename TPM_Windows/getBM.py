
### import used modules first

import os
from glob import glob
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np

### get analyzed sheet names

##  path_dat:list of path; sheet_names:list of string
def get_df_dict(path_data, sheet_names):
    df_dict = dict()
    n = 0
    for i, path in enumerate(path_data):
        for j, sheet_name in enumerate(sheet_names):
            df_dict[f'{n}'] = pd.read_excel(path, sheet_name=sheet_name, index_col=0)
            n += 1
    return df_dict
        

def get_analyzed_sheet_names():
    return ['BMx_sliding', 'BMy_sliding', 'BMx_fixing', 'BMy_fixing',
            'sx_sy', 'xy_ratio_sliding', 'xy_ratio_fixing', 'sx_over_sy_squared',
            'avg_attrs', 'std_attrs']

def get_attr(df_dict, column_name):
    data = []
    n = len(df_dict)
    for i in range(n):
        df = df_dict[f'{i}']
        data = np.append(data, np.array(df[column_name]))
        data = data.reshape(len(data),1)
    return data

def get_attr_mat(df_dict, column_names):
    attr = []
    for column_name in column_names:
        attr += [get_attr(df_dict, column_name=column_name)]
    attr = np.array(attr)
    return attr


def get_BM_avg(df_avg_attrs_dict):
    column_names = ['BMx_sliding', 'BMy_sliding', 'BMx_fixing', 'BMy_fixing']
    attr = get_attr_mat(df_avg_attrs_dict, column_names)
    BM_avg = np.mean(attr, axis=0)
    
    return BM_avg


def get_BM_raw(df_BM_dict):
    BM_raw = []
    n = len(df_BM_dict)
    for i in range(n):
        df_BM = df_BM_dict[f'{i}']
        BM_raw = np.append(BM_raw, np.array(df_BM))
    return BM_raw

root = tk.Tk()
root.withdraw()
path_folder = os.path.abspath(filedialog.askdirectory())

path_folders = glob(os.path.join(path_folder, '*'))
path_data = [glob(os.path.join(x, '*reshape_analyzed_selected.xlsx'))[0] for x in path_folders if glob(os.path.join(x, '*reshape_analyzed_selected.xlsx')) != []]


df_avg_attrs_dict = get_df_dict(path_data, sheet_names=['avg_attrs'])
df_BM_dict = get_df_dict(path_data, sheet_names=['BMx_fixing'])
df_sxsy_dict = get_df_dict(path_data, sheet_names=['sx_sy'])
# df_BMy_dict = get_df_dict(path_data, sheet_names='BMx_sliding')

sx_sy = get_attr(df_avg_attrs_dict, column_name='sx_sy')
BM_avg = get_BM_avg(df_avg_attrs_dict)

BM_raw = get_BM_raw(df_BM_dict)
sxsy_raw = get_BM_raw(df_sxsy_dict)
