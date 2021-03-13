
### import used modules first

from TPM.localization import select_folder
from glob import glob
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
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

### get analyzed sheet names

##  path_dat:list of path; sheet_names:list of string, axis=0(add vertically)
def get_df_dict(path_data, sheet_names, axis):
    df_dict = dict()
    for i, path in enumerate(path_data):
        for sheet_name in sheet_names:
            if i==0: ## initiate df_dict
                df = pd.read_excel(path, sheet_name=sheet_name)
                df_dict[f'{sheet_name}'] = df
            else: ## append df_dict
                df = pd.read_excel(path, sheet_name=sheet_name)
                df_dict[f'{sheet_name}'] = pd.concat([df_dict[f'{sheet_name}'], df], axis=axis)
    return df_dict

def get_analyzed_sheet_names():
    return ['BMx_sliding', 'BMy_sliding', 'BMx_fixing', 'BMy_fixing',
            'sx_sy', 'xy_ratio_sliding', 'xy_ratio_fixing', 'sx_over_sy_squared',
            'avg_attrs', 'std_attrs']

##  concatenate cetain attr from all df, output: 1D
def get_attr(df_dict, column_name):
    data = []
    n = len(df_dict)
    for i in range(n):
        df = df_dict[f'{i}']
        data = np.append(data, np.array(df[column_name]))
        data = data.reshape(len(data),1)
    return data


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
    data = np.array(data)
    data_nor = []
    for datum in data.T:
        mean = np.mean(datum)
        std = np.std(datum, ddof=1)
        datum_nor = (datum-mean)/std
        data_nor += [datum_nor]
    return np.nan_to_num(np.array(data_nor).T)


### get analyzed sheet names, add median
def get_analyzed_sheet_names():
    return ['BMx_sliding', 'BMy_sliding', 'BMx_fixing', 'BMy_fixing',
            'sx_sy', 'xy_ratio_sliding', 'xy_ratio_fixing', 'sx_over_sy_squared']

### get reshape sheet names
def get_reshape_sheet_names():
    return ['amplitude', 'sx', 'sy', 'x', 'y', 'theta_deg', 'offset', 'intensity', 'intensity_integral', 'ss_res']


def get_data_from_excel(path_folder, sheet_names, excel_name, axis):
    path_folders = glob(os.path.join(path_folder, '*'))
    path_data = [glob(os.path.join(x, '*'+excel_name))[0] for x in path_folders if
                 glob(os.path.join(x, '*'+excel_name)) != []]
    # sheet_names = ['med_attrs', 'std_attrs', 'avg_attrs']
    df_attrs_dict = get_df_dict(path_data, sheet_names, axis)
    return df_attrs_dict


excel_name = 'snapshot-fitresults_reshape_analyzed.xlsx'

path_folder = select_folder()
df_attrs_dict = get_data_from_excel(path_folder, sheet_names=['med_attrs', 'std_attrs', 'avg_attrs'], excel_name=excel_name, axis=0)
df_analyzed_dict = get_data_from_excel(path_folder, sheet_names=get_analyzed_sheet_names(), excel_name=excel_name, axis=1)


##  select statistical attributes for clustering analysis
select_columns = ['BMx_fixing', 'BMy_fixing', 'sx_sy']
sheet_names = ['med_attrs', 'std_attrs', 'avg_attrs']
df_select_attrs_dict = dict()
df_select_attrs_nor_dict = dict()
for sheet_name in sheet_names:
    df_select = df_attrs_dict[f'{sheet_name}'][select_columns]
    
    df_select_attrs_dict[f'{sheet_name}'] = df_attrs_dict[f'{sheet_name}'][select_columns]
    df_select_attrs_nor_dict[f'{sheet_name}'] = pd.DataFrame(data=normalize_data(df_select), columns=df_select.columns, index=df_select.index)


##  clustering
data = df_select_attrs_nor_dict['med_attrs']
pca = PCA(n_components=2)
result = pca.fit(data)
transform = result.transform(data)
X = transform

silhouette_array = []
BICs = []
n_clusters = np.arange(2,10)
for c in n_clusters:
    model = GaussianMixture(n_components=c)
    # model = Birch(threshold=0.005, n_clusters=c)
    model.fit(X)
    label = model.predict(X)
    silhouette_array += [silhouette_score(X, label)]
    BICs += [model.bic(X)]

# plt.figure()
# plt.plot(n_clusters, silhouette_array,'o')
# plt.title('silhouette')
# plt.figure()
# plt.plot(n_clusters, BICs,'o')
# plt.title('BIC')
# plt.xlabel('n_components')
# plt.ylabel('BIC')


n_components = n_clusters[np.argmin(BICs)]
model = GaussianMixture(n_components=n_components, tol=1e-5)
# model = Birch(threshold=0.05, n_clusters=5)
##  fit the model
model.fit(X)
##  assign a cluster to each example
label = model.predict(X)

beads_name = df_attrs_dict['med_attrs']['Unnamed: 0']
sx_sy = df_select_attrs_dict['med_attrs']['sx_sy']
S = [sx_sy[label==i] for i in range(n_components)]
S_name = [beads_name[label==i] for i in range(n_components)]
df_S_dict = dict()

for i, s in enumerate(S):
    df_S_dict[f'{i}'] = pd.DataFrame(data=s).set_index(S_name[i])
S_n_samples = [len(x) for x in S]
S_mean = [np.mean(x) for x in S]
S_std = [np.std(x, ddof=1) for x in S]
S_stat = np.array([S_n_samples, S_mean, S_std])
columns_stat = [f'DNA_{i}' for i in range(n_components)]
index_stat = ['n_samples', 'mean', 'std']
df_S_stat = pd.DataFrame(data=S_stat, index=index_stat, columns=columns_stat)

##  select analyzed data for saving
filename_time = get_date()
random_string = gen_random_code(3)
filename = 'statistics.xlsx'
sheet_names = [f'DNA_{i}' for i in range(n_components)]
writer = pd.ExcelWriter(os.path.join(path_folder, f'{filename_time}-{random_string}-{filename}'))
df_S_stat.to_excel(writer, sheet_name='statistics', index=True)
for i, sheet_name in enumerate(sheet_names):
    df_S_dict[f'{i}'].to_excel(writer, sheet_name=sheet_name, index=True)
writer.save()

