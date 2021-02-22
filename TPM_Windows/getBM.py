
### import used modules first

from localization import select_folder
from glob import glob
import tkinter as tk
from tkinter import filedialog
import random
import string
import numpy as np
import os
import datetime
import pandas as pd
import scipy.linalg as la
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D


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

def get_all_attrs(df_dict):
    data = np.array(df_dict['0'])
    n = len(df_dict)
    for i in range(n-1):
        df = df_dict[f'{i+1}']
        data = np.append(data, np.array(df), axis=0)
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

##  add 2n-word random texts(n-word number and n-word letter)
def gen_random_code(n):
    digits = "".join([random.choice(string.digits) for i in range(n)])
    chars = "".join([random.choice(string.ascii_letters) for i in range(n)])
    return digits + chars

### getting date
def get_date():
    filename_time = datetime.datetime.today().strftime('%Y-%m-%d')  # yy-mm-dd
    return filename_time

### normalize to mean = 0, std = 1
def normalize_data(data):
    data_nor = []
    for datum in data.T:
        mean = np.mean(datum)
        std = np.std(datum, ddof=1)
        datum_nor = (datum-mean)/std
        data_nor += [datum_nor]
    return np.nan_to_num(np.array(data_nor).T)

### sort eignvectors by eigen value, vectors is a matrix(column as eigenvector), values is array
def sorteigen(values, vectors):
    i = np.argsort(values)[::-1]
    output_values = np.array([values[x] for x in i]).reshape(len(values),1)
    output_vectors = np.array([vectors[:,x] for x in i])
    return output_values.real, output_vectors # eigenvaluses must be real number because cov matrix is positive semi-definite

### get label data
def get_label_data(path_folder):
    all_attrs_nor_selected, all_med_attrs_nor_s = get_all_excel_data(path_folder, 'reshape_analyzed_selected.xlsx')
    all_attrs_nor_selected = np.append(all_attrs_nor_selected, np.ones((all_attrs_nor_selected.shape[0], 1)), axis=1)
    all_med_attrs_nor_s = np.append(all_med_attrs_nor_s, np.ones((all_med_attrs_nor_s.shape[0], 1)), axis=1)
    all_attrs_nor_removed, all_med_attrs_nor_r = get_all_excel_data(path_folder, 'reshape_analyzed_removed.xlsx')
    all_attrs_nor_removed = np.append(all_attrs_nor_removed, np.zeros((all_attrs_nor_removed.shape[0], 1)), axis=1)
    all_med_attrs_nor_r = np.append(all_med_attrs_nor_r, np.ones((all_med_attrs_nor_r.shape[0], 1)), axis=1)

    # return all_attrs_nor_selected, all_attrs_nor_removed
    all_attrs_nor_label = np.append(all_attrs_nor_selected, all_attrs_nor_removed, axis=0)
    all_med_attrs_nor_label = np.append(all_med_attrs_nor_s, all_med_attrs_nor_r, axis=0)
    return all_attrs_nor_label, all_med_attrs_nor_label

##  reduce dimension[BM, xy_ratio, ]
def reduce_dim(med_attrs):
    n = med_attrs.shape[0]
    selected_attrs = []
    BM = np.mean(med_attrs[:, 0:4], axis=1).reshape((n,1))
    xy_ratio = np.mean(med_attrs[:, 5:8], axis=1).reshape((n,1))
    s = np.mean(med_attrs[:, 9:11], axis=1).reshape((n,1))
    intensity_integral = med_attrs[:, -2].reshape((n,1))
    ss_res = med_attrs[:, -1].reshape((n,1))
    pre_append_data = [BM, xy_ratio, s, intensity_integral, ss_res]

    selected_attrs = np.array(pre_append_data).reshape(5,n).T
    return selected_attrs

# ##  append all data
# def append_operator(*args, axis=1):
#     n = np.shape(args[0])[0]
#     data = np.empty((n,1))
#     for arg in args:
#         data = np.append(data, arg, axis=axis)
#     return data

### output = [med(18), std(18), avg(18+1)]
def get_all_excel_data(path_folder, file_type='reshape_analyzed.xlsx'):
    # path_folder = select_folder()
    path_folders = glob(os.path.join(path_folder, '*'))
    path_data = [glob(os.path.join(x, '*'+file_type))[0] for x in path_folders if
                 glob(os.path.join(x, '*'+file_type)) != []]
    df_avg_attrs_dict = get_df_dict(path_data, sheet_names=['avg_attrs'])
    df_med_attrs_dict = get_df_dict(path_data, sheet_names=['med_attrs'])
    df_std_attrs_dict = get_df_dict(path_data, sheet_names=['std_attrs'])

    all_med_attrs = get_all_attrs(df_med_attrs_dict)
    all_med_attrs = reduce_dim(all_med_attrs)
    all_std_attrs = get_all_attrs(df_std_attrs_dict)
    all_std_attrs = reduce_dim(all_std_attrs)
    all_avg_attrs = get_all_attrs(df_avg_attrs_dict)
    all_bead_radius = all_avg_attrs[:,-1].reshape((all_avg_attrs.shape[0],1))
    all_avg_attrs = reduce_dim(all_avg_attrs[:,:-1])


    all_attrs = np.append(all_med_attrs, all_std_attrs, axis=1)
    all_attrs = np.append(all_attrs, all_avg_attrs, axis=1)
    all_attrs = np.append(all_attrs, all_bead_radius, axis=1)

    all_attrs_nor = normalize_data(all_attrs)
    all_med_attrs_nor = normalize_data(all_med_attrs)

    return all_attrs_nor, all_med_attrs_nor

def get_eigens(all_attrs_nor):
    covariance_matrix = np.matmul(np.transpose(all_attrs_nor), all_attrs_nor) / (all_attrs_nor.shape[0] - 1)
    eigen_values, eigen_vectors = la.eig(covariance_matrix)
    eigen_values, eigen_vectors = sorteigen(eigen_values, eigen_vectors)
    eigen_values_ps = eigen_values/sum(eigen_values)
    return eigen_values_ps, eigen_values, eigen_vectors


# def get_columns(bead_number):
#     columns = [f'bead_{i}' for i in range(bead_number)]
#     return np.array(columns)


# root = tk.Tk()
# root.withdraw()
# path_folder = os.path.abspath(filedialog.askdirectory())
path_folder = select_folder()
path_folders = glob(os.path.join(path_folder, '*'))
path_data = [glob(os.path.join(x, '*reshape_analyzed.xlsx'))[0] for x in path_folders if glob(os.path.join(x, '*reshape_analyzed_selected.xlsx')) != []]

# ### get data
# df_avg_attrs_dict = get_df_dict(path_data, sheet_names=['avg_attrs'])
# df_med_attrs_dict = get_df_dict(path_data, sheet_names=['med_attrs'])
# df_std_attrs_dict = get_df_dict(path_data, sheet_names=['std_attrs'])
#
# all_med_attrs = get_all_attrs(df_med_attrs_dict)
# all_std_attrs = get_all_attrs(df_std_attrs_dict)
# all_avg_attrs = get_all_attrs(df_avg_attrs_dict)
#
# all_med_attrs = reduce_dim(all_med_attrs)
# all_std_attrs = reduce_dim(all_std_attrs)
# all_avg_attrs = reduce_dim(all_avg_attrs)
#
# all_attrs = np.append(all_med_attrs, all_std_attrs, axis=1)
# all_attrs = np.append(all_attrs, all_avg_attrs, axis=1)
# all_attrs_nor = normalize_data(all_attrs)


all_attrs_nor_label, all_med_attrs_nor_label = get_label_data(path_folder)
label = all_attrs_nor_label[:, -1].reshape(all_attrs_nor_label.shape[0],1)

all_attrs_nor = all_attrs_nor_label[:, :-1]
all_med_attrs_nor = all_med_attrs_nor_label[:, :-1]

eigen_values_ps, eigen_values, eigen_vectors = get_eigens(all_attrs_nor)

##  use toolkit
pca = PCA(n_components=5)
result = pca.fit(all_med_attrs_nor)
transform = result.transform(all_med_attrs_nor)
x_all = transform[:,1]
y_all = transform[:,2]
# z_all = transform[:,3]

z_all = all_med_attrs_nor_label[:, 0]
for x,y,l in zip(x_all, y_all, label):
    if l == 1: # selected color is red
        c='r'
    else:
        c='b' # removed color is blue
    plt.plot(x ,y ,c+'o')


fig = plt.figure()
ax = Axes3D(fig)
for x, y, z, l in zip(x_all, y_all, z_all, label):
    if l == 1:  # selected color is red
        c = 'r'
    else:
        c = 'b'  # removed color is blue
    ax.scatter(x, y, z, c=c)
plt.show()



# 設定ax為散佈圖





# eigen_values_ps_s, eigen_values_s, eigen_vectors_s = get_eigens(all_attrs_nor_selected)
# eigen_values_ps_r, eigen_values_r, eigen_vectors_r = get_eigens(all_attrs_nor_removed)

# ###  for PCA analysis
# ##  get covariance matrix
# covariance_matrix = np.matmul(np.transpose(all_attrs_nor),all_attrs_nor)/(all_attrs_nor.shape[0]-1)
# ##  get eigenvalue of covariance_matrix
# eigen_values, eigen_vectors = la.eig(covariance_matrix)
# eigen_values, eigen_vectors = sorteigen(eigen_values, eigen_vectors)
# eigen_values_ps = eigen_values/sum(eigen_values)
#
#
# ##  use toolkit
# pca = PCA(n_components=3)
# result = pca.fit(all_attrs_nor)
# transform = result.transform(all_attrs_nor)
# x = transform[:,0]
# y = transform[:,1]
# z = transform[:,2]
#
#
# plt.plot(transform[:,0], transform[:,1], 'o')
# plt.plot(transform[:,0], transform[:,2], 'o')
# plt.plot(transform[:,1], transform[:,2], 'o')
#
#
# # 製作figure
# fig = plt.figure()
#
# ax = Axes3D(fig)
#
# # 設定ax為散佈圖
# ax.scatter(x, y, z)
#
# plt.show()
#
# df_BM_dict = get_df_dict(path_data, sheet_names=['BMx_fixing'])
# df_sxsy_dict = get_df_dict(path_data, sheet_names=['sx_sy'])
#
# sx_sy_avg = get_attr(df_avg_attrs_dict, column_name='sx_sy')
# BM_avg = get_BM_avg(df_avg_attrs_dict)
# bead_radius = get_attr(df_avg_attrs_dict, column_name='bead_radius')
# sx_sy_med = get_attr(df_med_attrs_dict, column_name='sx_sy')
# BM_med = get_BM_avg(df_med_attrs_dict)
#
# BM_raw = get_BM_raw(df_BM_dict)
# sxsy_raw = get_BM_raw(df_sxsy_dict)
#
# results = np.array([BM_med, sx_sy_med, BM_avg, sx_sy_avg, bead_radius]).reshape(5,len(BM_med)).T
# sheet_names = ['BM_median', 'sxsy_mediad', 'BM_avg', 'sxsy_avg', 'bead_radius']
# columns = ['BM_median', 'sxsy_mediad', 'BM_avg', 'sxsy_avg', 'bead_radius']
#
#
# filename_time = get_date()
# random_string = gen_random_code(3)
# filename = 'analyze_all_folders.xlsx'
#
# df_results = pd.DataFrame(data=results, columns=columns)
# df_results.to_excel(os.path.join(path_folder, f'{filename_time}-{random_string}-{filename}'), index=True)




#
# writer = pd.ExcelWriter(os.path.join(path_folder, f'{filename_time}-{random_string}-{filename}'))
# for sheet_name, data in zip(sheet_names):
#     df_results = pd.DataFrame(data=data, columns=sheet_name)
#     df_results.to_excel(writer, sheet_name=sheet_name, index=True)
# writer.save()