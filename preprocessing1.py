# -*- coding: utf-8 -*-
# Author: Antoine DELPLACE
# Last update: 17/01/2020
"""
从 Netflow 文件中提取窗口相关数据的预处理程序
处理一个网络流量数据集，为机器学习分析（特别是检测僵尸网络活动）做准备

Parameters
----------
window_width  : window width in seconds
window_stride : window stride in seconds
data          : pandas DataFrame of the Netflow file

Return
----------
Create 3 output files:
- data_window_botnetx.h5         : DataFrame with the extracted data: Sport, DstAddr, Dport,
                                   Dur (sum, mean, std, max, median), TotBytes (sum, mean, std, max, median),
                                   SrcBytes (sum, mean, std, max, median)
- data_window_botnetx_id.npy     : Numpy array containing SrcAddr
- data_window_botnetx_labels.npy : Numpy array containing Label
"""

import pandas as pd
import numpy as np
import datetime
import h5py

from scipy.stats import mode

# 定义时间窗口的参数
window_width = 120  # seconds
window_stride = 60  # seconds

# 导入数据集
print("Import data")
data = pd.read_csv("capture20110811.binetflow")
# with pd.option_context('display.max_rows', None, 'display.max_columns', 15):
#    print(data.shape)
#    print(data.head())
#    print(data.dtypes)
print("Preprocessing")


# 通过减去均值并除以标准差来标准化指定列
def normalize_column(dt, column):
    mean = dt[column].mean()
    std = dt[column].std()
    print(mean, std)
    dt[column] = (dt[column] - mean) / std


"""
数据准备
将 StartTime 转换为数值格式，并归一化为秒。
datetime_start 存储数据集中最早的开始时间。
计算 Window_lower 和 Window_upper_excl 以确定每条记录所属的窗口范围，确保下限至少为0。
删除原始的 StartTime 列，因为它已不再需要。
"""
data['StartTime'] = pd.to_datetime(data['StartTime']).astype(np.int64) * 1e-9
datetime_start = data['StartTime'].min()
data['Window_lower'] = (data['StartTime'] - datetime_start - window_width) / window_stride + 1
data['Window_lower'].clip(lower=0, inplace=True)
data['Window_upper_excl'] = (data['StartTime'] - datetime_start) / window_stride + 1
data = data.astype({"Window_lower": int, "Window_upper_excl": int})
data.drop('StartTime', axis=1, inplace=True)

# 使用 pd.factorize 将 Label 列编码为整数
data['Label'], labels = pd.factorize(data['Label'].str.slice(0, 15))
# print(data.dtypes)

# 创建时间窗口
X = pd.DataFrame()
nb_windows = data['Window_upper_excl'].max()
print(nb_windows)

for i in range(0, nb_windows):
    gb = data.loc[(data['Window_lower'] <= i) & (data['Window_upper_excl'] > i)].groupby('SrcAddr')
    # 将聚合数据附加到 X，并将每条记录与当前窗口 ID 关联。
    X = X.append(gb.size().to_frame(name='counts').join(gb.agg({'Sport': 'nunique',
                                                                'DstAddr': 'nunique',
                                                                'Dport': 'nunique',
                                                                'Dur': ['sum', 'mean', 'std', 'max', 'median'],
                                                                'TotBytes': ['sum', 'mean', 'std', 'max', 'median'],
                                                                'SrcBytes': ['sum', 'mean', 'std', 'max', 'median'],
                                                                'Label': lambda x: mode(x)[0]})).reset_index().assign(
        window_id=i))
    print(X.shape)

del (data)

# 通过用下划线连接元组来将多级列名转换为单级
X.columns = ["_".join(x) if isinstance(x, tuple) else x for x in X.columns.ravel()]
# print(X.columns.values)

# 所有的 NaN 值变为 -1
X.fillna(-1, inplace=True)

# print(X.columns.values)
columns_to_normalize = list(X.columns.values)
columns_to_normalize.remove('SrcAddr')
columns_to_normalize.remove('Label_<lambda>')
columns_to_normalize.remove('window_id')

normalize_column(X, columns_to_normalize)

# 将处理后的数据（不包括 SrcAddr）保存到 HDF5 文件
with pd.option_context('display.max_rows', 10, 'display.max_columns', 22):
    print(X.shape)
    print(X)
    print(X.dtypes)

# with pd.option_context('display.max_rows', 10, 'display.max_columns', 20):
#    print(X.loc[X['Label'] != 0])

X.drop('SrcAddr', axis=1).to_hdf('data_window_botnet.h5', key="data", mode="w")
np.save("data_window_botnet_id.npy", X['SrcAddr'])
np.save("data_window_botnet_labels.npy", labels)
