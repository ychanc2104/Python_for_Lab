
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
from sklearn.cluster import Birch, KMeans
from sklearn.metrics import silhouette_score




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
# sx_sy = np.array(df['med_sx'] * df['med_sy'])
sx_sy = np.array(df['sx_sy'])

selected_attrs = np.array(df_nor)

##  get PCA data

# keys_list = ['amplitude', 'offset', 'intensity', 'intensity_integral', 'ss_res']
# selected_attrs = select_attrs(all_attrs_nor, keys_list)

##  use own-make function
# eigen_values_ps, eigen_values, eigen_vectors = get_eigens(selected_attrs)

##  use toolkit
pca = PCA(n_components=3)
result = pca.fit(selected_attrs)
transform = result.transform(selected_attrs)
X = transform


# X = selected_attrs
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

plt.figure()
plt.plot(n_clusters, silhouette_array,'o')
plt.title('silhouette')
plt.figure()
plt.plot(n_clusters, BICs,'o')
plt.title('BIC')


n_components = n_clusters[np.argmin(BICs)]
model = GaussianMixture(n_components=n_components, tol=1e-5)
# model = Birch(threshold=0.05, n_clusters=5)
# fit the model
model.fit(X)
# assign a cluster to each example
label = model.predict(X)


# s0 = sx_sy[label==0]
# s1 = sx_sy[label==1]
    

s0 = sx_sy[label==0]
s1 = sx_sy[label==1]
s2 = sx_sy[label==2]
s3 = sx_sy[label==3]
s4 = sx_sy[label==4]
s5 = sx_sy[label==5]


# retrieve unique clusters
clusters = unique(label)
# # create scatter plot for samples from each cluster
fig = plt.figure()
ax = Axes3D(fig)
for cluster in clusters:
 	# get row indexes for samples with this cluster
 	row_ix = where(label == cluster)
 	# create scatter of these samples
 	ax.scatter(X[row_ix, 0], X[row_ix, 1], X[row_ix, 2])
    # ax.scatter(X[row_ix, 0], X[row_ix, 1])

# show the plot
ax.set_xlabel('PC0')
ax.set_ylabel('PC1')
ax.set_zlabel('PC2')

# plt.show()
