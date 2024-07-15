# -*- coding: utf-8 -*-
# Author: Antoine DELPLACE
# Last update: 17/01/2020
"""
使用 PCA 和 t-SNE 的降维算法
通过 t-SNE 和 PCA 方法对数据集中的特征进行降维，并可视化这些特征在二维空间中的分布，
可以帮助识别不同类别样本的分布特性和潜在的聚类结构

Parameters
----------
data_window.h5         : extracted data from preprocessing1.py
data_window3.h5        : extracted data from preprocessing2.py
data_window_labels.npy : label numpy array from preprocessing1.py

Return
----------
利用 PCA 或 t-SNE 绘制数据的二维表示
"""

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import h5py

from sklearn import model_selection, manifold, decomposition

print("Import data")

# 读取数据并合并
X = pd.read_hdf('data_window.h5', key='data')
X.reset_index(drop=True, inplace=True)

X2 = pd.read_hdf('data_window3.h5', key='data')
X2.reset_index(drop=True, inplace=True)

X = X.join(X2)

X.drop('window_id', axis=1, inplace=True)

y = X['Label_<lambda>']
X.drop('Label_<lambda>', axis=1, inplace=True)

labels = np.load("data_window_labels.npy")

print(X.columns.values)
print(labels)
print(np.where(labels == 'flow=From-Botne')[0][0])

# 创建二元标签，打印标签的唯一值及其计数
y_bin6 = y==np.where(labels == 'flow=From-Botne')[0][0]
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y_bin6, test_size=0.33, random_state=123456)
print("y", np.unique(y, return_counts=True))
print("y_train", np.unique(X_train, return_counts=True))
print("y_test", np.unique(y_test, return_counts=True))

# 使用 t-SNE 降维方法，将选择的特征降到二维空间，并打印降维结果的散点图
print("t-SNE")
clf = manifold.TSNE(n_components=2, random_state=123456)
clf.fit(X[['Dport_nunique', 'TotBytes_sum', 'Dur_sum', 'Dur_mean', 'TotBytes_std']])

print(clf.embedding_)

y_plot = np.where(y_bin6 == True)[0]
print(len(y_plot))

y_plot2 = np.random.choice(np.where(y_bin6 == False)[0], size=len(y_plot)*100, replace=False)
print(len(y_plot2))

index = list(y_plot)+list(y_plot2)
print(len(index))

plt.scatter(clf.embedding_[index, 0], clf.embedding_[index, 1], c=y[index])
plt.colorbar()
plt.show()


# 使用 PCA 降维方法，将选择的特征降到二维空间，并打印降维结果的散点图
print("PCA")
clf = decomposition.PCA(n_components=2, random_state=123456)
clf.fit(X[['Dport_nunique', 'TotBytes_sum', 'Dur_sum', 'Dur_mean', 'TotBytes_std']].transpose())

print(clf.components_)
print(clf.explained_variance_ratio_)

y_plot = np.where(y_bin6 == True)[0]
print(len(y_plot))

y_plot2 = np.random.choice(np.where(y_bin6 == False)[0], size=len(y_plot)*100, replace=False)
print(len(y_plot2))

index = list(y_plot)+list(y_plot2)
print(len(index))

plt.scatter(clf.components_[0, index], clf.components_[1, index], c=y[index])
plt.colorbar()
plt.show()


