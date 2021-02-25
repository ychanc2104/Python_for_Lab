
### import used modules first


from localization import select_folder
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
from sklearn.cluster import Birch




def get_eigens(all_attrs_nor):
    covariance_matrix = np.matmul(np.transpose(all_attrs_nor), all_attrs_nor) / (all_attrs_nor.shape[0] - 1)
    eigen_values, eigen_vectors = la.eig(covariance_matrix)
    eigen_values, eigen_vectors = sorteigen(eigen_values, eigen_vectors)
    eigen_values_ps = eigen_values/sum(eigen_values)
    return eigen_values_ps, eigen_values, eigen_vectors

### sort eignvectors by eigen value, vectors is a matrix(column as eigenvector), values is array
def sorteigen(values, vectors):
    i = np.argsort(values)[::-1]
    output_values = np.array([values[x] for x in i]).reshape(len(values),1)
    output_vectors = np.array([vectors[:,x] for x in i])
    return output_values.real, output_vectors # eigenvaluses must be real number because cov matrix is positive semi-definite

##  reduce dimension[BM, xy_ratio, ]
def select_attrs(all_attrs, keys_list):
    columns = ['BMx_sliding', 'BMy_sliding', 'BMx_fixing', 'BMy_fixing',
                'sx_sy', 'xy_ratio_sliding', 'xy_ratio_fixing', 'sx_over_sy_squared',
                'amplitude', 'sx', 'sy', 'x', 'y', 'theta_deg', 'offset', 'intensity',
                'intensity_integral', 'ss_res']
    n_features = len(columns)
    attrs_dict = dict()
    for value, key in enumerate(columns):
        attrs_dict[key] = value
    index_select = [attrs_dict[x] for x in keys_list] + [attrs_dict[x]+n_features for x in keys_list] + [attrs_dict[x]+2*n_features for x in keys_list]
    selected_attrs = all_attrs[:, index_select]
    return selected_attrs



path_folder = select_folder()
path_data = os.path.abspath(glob(os.path.join(path_folder, '*collecting_features.xlsx'))[0])

df_nor = pd.read_excel(path_data, sheet_name='normalized', index_col=0)
df = pd.read_excel(path_data, sheet_name='original', index_col=0)
sx_sy = df['med_sx'] * df['med_sy']

selected_attrs = np.array(df_nor)

##  get PCA data

# keys_list = ['amplitude', 'offset', 'intensity', 'intensity_integral', 'ss_res']
# selected_attrs = select_attrs(all_attrs_nor, keys_list)

##  use own-make function
eigen_values_ps, eigen_values, eigen_vectors = get_eigens(selected_attrs)

##  use toolkit
pca = PCA(n_components=3)
result = pca.fit(selected_attrs)
transform = result.transform(selected_attrs)


X = transform
# model = GaussianMixture(n_components=3)
model = Birch(threshold=0.05, n_clusters=3)
# fit the model
model.fit(X)
# assign a cluster to each example
label = model.predict(X)
s0 = sx_sy[label==0]
s1 = sx_sy[label==1]
s2 = sx_sy[label==2]

# retrieve unique clusters
clusters = unique(label)
# create scatter plot for samples from each cluster
for cluster in clusters:
	# get row indexes for samples with this cluster
	row_ix = where(label == cluster)
	# create scatter of these samples
	plt.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
plt.show()
