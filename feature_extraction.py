# -*- coding: utf-8 -*-
# Author: Antoine DELPLACE
# Last update: 17/01/2020
"""
使用不同的嵌入方法提取相关特征：
- Lasso 和 Ridge Logistic 回归
- 带有递归特征消除 (RFE) 的支持向量机 (SVM) 方法

Parameters
----------
data_window.h5         : extracted data from preprocessing1.py
data_window3.h5        : extracted data from preprocessing2.py
data_window_labels.npy : label numpy array from preprocessing1.py

Return
----------
打印不同方法的结果（准确率、召回率、f1）,绘制不同提取的图表
"""

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import h5py
from sklearn import model_selection, feature_selection, linear_model, metrics

# 通过逻辑回归模型对网络流量数据进行二分类，并评估模型性能
print("Import data")

# 读取两个数据集 data_window.h5 和 data_window3.h5
X = pd.read_hdf('data_window.h5', key='data')
X.reset_index(drop=True, inplace=True)

X2 = pd.read_hdf('data_window3.h5', key='data')
X2.reset_index(drop=True, inplace=True)

# 合并数据集
X = X.join(X2)

# 删除 window_id 列和 Label_<lambda> 列
X.drop('window_id', axis=1, inplace=True)
y = X['Label_<lambda>']
X.drop('Label_<lambda>', axis=1, inplace=True)

labels = np.load("data_window_labels.npy")

# print(X)
# print(y)
print(X.columns.values)
print(labels)

# 将标签 y 转换为二分类标签 y_bin6，其中标签为6的样本为 True，其余为 False
y_bin6 = y == 6
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y_bin6, test_size=0.33, random_state=123456)
# y_train_bin6 = y_train==6
# y_test_bin6 = y_test==6

print("y", np.unique(y, return_counts=True))
print("y_train", np.unique(y_train, return_counts=True))
print("y_test", np.unique(y_test, return_counts=True))

# 逻辑回归模型训练和评估
print("Logistic Regression")

clf = linear_model.LogisticRegression(penalty='l2', C=1.0, random_state=123456, multi_class="auto", class_weight=None,
                                      solver="lbfgs", max_iter=1000, verbose=1)
# 在训练集上训练模型，并打印模型系数和截距。
clf.fit(X_train, y_train)
# print(clf.classes_)
print(clf.coef_)
print(clf.intercept_)

# 在测试集上进行预测，并计算和打印模型的平衡准确率、精确度、召回率、F1分数和支持度。
y_pred = clf.predict(X_test)
# y_pred_bin6 = y_pred==6
# print(clf.predict_proba(X_test))
print("accuracy score = ", metrics.balanced_accuracy_score(y_test, y_pred))
precision, recall, fbeta_score, support = metrics.precision_recall_fscore_support(y_test, y_pred)
print("precision = ", precision[1])
print("recall = ", recall[1])
print("fbeta_score = ", fbeta_score[1])
print("support = ", support[1])

# 使用不同的类权重
clf = linear_model.LogisticRegression(penalty='l2', C=1.0, random_state=123456, multi_class="auto",
                                      class_weight='balanced', solver="lbfgs", max_iter=1000, verbose=1)
clf.fit(X_train, y_train)
# print(clf.classes_)
print(clf.coef_)
print(clf.intercept_)

y_pred = clf.predict(X_test)
# y_pred_bin6 = y_pred==6
# print(clf.predict_proba(X_test))
print("accuracy score = ", metrics.balanced_accuracy_score(y_test, y_pred))
precision, recall, fbeta_score, support = metrics.precision_recall_fscore_support(y_test, y_pred)
print("precision = ", precision[1])
print("recall = ", recall[1])
print("fbeta_score = ", fbeta_score[1])
print("support = ", support[1])

# 使用自定义的类权重
clf = linear_model.LogisticRegression(penalty='l2', C=1.0, random_state=123456, multi_class="auto",
                                      class_weight={0: 0.5, 1: 0.5}, solver="lbfgs", max_iter=1000, verbose=1)
clf.fit(X_train, y_train)
# print(clf.classes_)
print(clf.coef_)
print(clf.intercept_)

y_pred = clf.predict(X_test)
# y_pred_bin6 = y_pred==6
# print(clf.predict_proba(X_test))
print("accuracy score = ", metrics.balanced_accuracy_score(y_test, y_pred))
precision, recall, fbeta_score, support = metrics.precision_recall_fscore_support(y_test, y_pred)
print("precision = ", precision[1])
print("recall = ", recall[1])
print("fbeta_score = ", fbeta_score[1])
print("support = ", support[1])

# main problems:
# with class_weight='balanced', super high recall but very low precision
# without, high precision but very low recall
# accuracy score not a good metric (even balanced_accuracy)

print("Logistic Regression Cross Validation")


# 使用交叉验证来评估和优化逻辑回归模型的性能
def apply_logreg_cross_validation(X, y,
                                  svc_args={'penalty': 'l2', 'C': 1.0, 'random_state': 123456, 'multi_class': "auto",
                                            'class_weight': None, 'solver': "lbfgs", 'max_iter': 1000, 'verbose': 1}):
    # 定义验证函数
    clf = linear_model.LogisticRegression(**svc_args)
    cv = model_selection.ShuffleSplit(n_splits=10, test_size=0.1, random_state=123456)
    # 计算模型的精确度、召回率和F1分数，并返回这些指标的平均值
    scores = model_selection.cross_validate(clf, X, y, cv=cv, scoring=['precision', 'recall', 'f1'],
                                            return_train_score=True)
    print(scores)
    return [np.mean(scores['test_precision']), np.mean(scores['test_recall']), np.mean(scores['test_f1'])]

# 调整类权重
tab_class_weight = np.linspace(0, 0.1, 10)
print(tab_class_weight)

tab_score = np.array([apply_logreg_cross_validation(X_train, y_train,
                                                    {'penalty': 'l2', 'C': 1.0, 'random_state': 123456,
                                                     'multi_class': "auto", 'class_weight': {0: w, 1: 1 - w},
                                                     'solver': "lbfgs", 'max_iter': 1000, 'verbose': 0}) for w in
                      tab_class_weight])
print(tab_score)

plt.plot(tab_class_weight, tab_score[:, 0])
plt.plot(tab_class_weight, tab_score[:, 1])
plt.plot(tab_class_weight, tab_score[:, 2])
plt.legend(["test_precision", "test_recall", "test_f1"])
plt.xlabel("Botnet class weight")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("cross_validation_class_weight.pdf", format="pdf")
plt.show()


# Results: class_weight_best = 0.044

# 调整正则化参数 C
def apply_logreg_cross_validation_coeff(X, y, svc_args={'penalty': 'l2', 'C': 1.0, 'random_state': 123456,
                                                        'multi_class': "auto", 'class_weight': None, 'solver': "lbfgs",
                                                        'max_iter': 1000, 'verbose': 1}):
    clf = linear_model.LogisticRegression(**svc_args)
    # cv = model_selection.ShuffleSplit(n_splits=10, test_size=0.1, random_state=123456) #for l2
    cv = model_selection.ShuffleSplit(n_splits=3, test_size=0.1, random_state=123456)  # for l1
    scores = model_selection.cross_validate(clf, X, y, cv=cv, scoring=['precision', 'recall', 'f1'],
                                            return_train_score=True, return_estimator=True)
    print(scores)
    return [np.mean(scores['test_precision']), np.mean(scores['test_recall']), np.mean(scores['test_f1']),
            np.mean([model.coef_[0] for model in scores['estimator']], axis=0)]


tab_C = np.logspace(-2, 6, 9)
tab_logC = np.log10(tab_C)
print(tab_C)
print(tab_logC)

tab_score = np.array([apply_logreg_cross_validation_coeff(X_train, y_train,
                                                          {'penalty': 'l2', 'C': C, 'random_state': 123456,
                                                           'multi_class': "auto",
                                                           'class_weight': {0: 0.044, 1: 1 - 0.044}, 'solver': "lbfgs",
                                                           'max_iter': 1000, 'verbose': 0}) for C in tab_C])
print(tab_score)

plt.plot(tab_logC, tab_score[:, 0])
plt.plot(tab_logC, tab_score[:, 1])
plt.plot(tab_logC, tab_score[:, 2])
plt.legend(["test_precision", "test_recall", "test_f1"])
plt.xlabel("log(C)")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("cross_validation_C.pdf", format="pdf")
plt.show()

matrix_coeff = np.stack(tab_score[:, 3], axis=0)
print(matrix_coeff)
print(matrix_coeff.shape)

ax = plt.subplot(111)
NUM_COLORS = matrix_coeff.shape[1]
LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
NUM_STYLES = len(LINE_STYLES)
cm = plt.get_cmap('Set1')

for i in range(0, matrix_coeff.shape[1]):
    lines = ax.plot(tab_logC, matrix_coeff[:, i])
    lines[0].set_color(cm(i // NUM_STYLES))
    lines[0].set_linestyle(LINE_STYLES[i % NUM_STYLES])

plt.xlabel("log(C)")
plt.xticks(label=np.log(tab_C))

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])
ax.legend(np.arange(0, matrix_coeff.shape[1]), loc='center left', bbox_to_anchor=(1, 0.5))

plt.savefig("cross_validation_C_coeff.pdf", format="pdf")
plt.show()

print(matrix_coeff)

tab_C = np.linspace(550, 1000, 10)
print(tab_C)

tab_score = np.array([apply_logreg_cross_validation_coeff(X_train, y_train,
                                                          {'penalty': 'l2', 'C': C, 'random_state': 123456,
                                                           'multi_class': "auto",
                                                           'class_weight': {0: 0.044, 1: 1 - 0.044}, 'solver': "lbfgs",
                                                           'max_iter': 1000, 'verbose': 0}) for C in tab_C])
print(tab_score)

plt.plot(tab_C, tab_score[:, 0])
plt.plot(tab_C, tab_score[:, 1])
plt.plot(tab_C, tab_score[:, 2])
plt.legend(["test_precision", "test_recall", "test_f1"])
plt.xlabel("C")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("cross_validation_C.pdf", format="pdf")
plt.show()

matrix_coeff = np.stack(tab_score[:, 3], axis=0)
print(matrix_coeff)
print(matrix_coeff.shape)

ax = plt.subplot(111)
NUM_COLORS = matrix_coeff.shape[1]
LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
NUM_STYLES = len(LINE_STYLES)
cm = plt.get_cmap('Set1')

for i in range(0, matrix_coeff.shape[1]):
    lines = ax.plot(tab_C, matrix_coeff[:, i])
    lines[0].set_color(cm(i // NUM_STYLES))
    lines[0].set_linestyle(LINE_STYLES[i % NUM_STYLES])

plt.xlabel("C")

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])
ax.legend(np.arange(0, matrix_coeff.shape[1]), loc='center left', bbox_to_anchor=(1, 0.5))

plt.savefig("cross_validation_C_coeff.pdf", format="pdf")
plt.show()

tab_C = np.linspace(50, 1000, 20)
print(tab_C)

print(tab_score)

plt.plot(tab_C, tab_score[:, 0])
plt.plot(tab_C, tab_score[:, 1])
plt.plot(tab_C, tab_score[:, 2])
plt.legend(["test_precision", "test_recall", "test_f1"], loc='upper right', bbox_to_anchor=(1, 0.9))
plt.xlabel("C")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("cross_validation_C.pdf", format="pdf")
plt.show()

matrix_coeff = np.stack(tab_score[:, 3], axis=0)
print(matrix_coeff)
print(matrix_coeff.shape)

ax = plt.subplot(111)
NUM_COLORS = matrix_coeff.shape[1]
LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
NUM_STYLES = len(LINE_STYLES)
cm = plt.get_cmap('Set1')

for i in range(0, matrix_coeff.shape[1]):
    lines = ax.plot(tab_C, matrix_coeff[:, i])
    lines[0].set_color(cm(i // NUM_STYLES))
    lines[0].set_linestyle(LINE_STYLES[i % NUM_STYLES])

plt.xlabel("C")

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])
ax.legend(np.arange(0, matrix_coeff.shape[1]), loc='center left', bbox_to_anchor=(1, 0.5))

plt.savefig("cross_validation_C_coeff.pdf", format="pdf")
plt.show()

# tab_C = np.logspace(-2, 6, 9)
tab_C = [1e6]
tab_logC = np.log10(tab_C)
print(tab_C)
print(tab_logC)

tab_score = np.array([apply_logreg_cross_validation_coeff(X_train, y_train,
                                                          {'penalty': 'l1', 'C': C, 'random_state': 123456,
                                                           'multi_class': "auto",
                                                           'class_weight': {0: 0.044, 1: 1 - 0.044},
                                                           'solver': "liblinear", 'max_iter': 1000, 'verbose': 1}) for C
                      in tab_C])
print(tab_score)

plt.plot(tab_logC, tab_score[:, 0])
plt.plot(tab_logC, tab_score[:, 1])
plt.plot(tab_logC, tab_score[:, 2])
plt.legend(["test_precision", "test_recall", "test_f1"])
plt.xlabel("log(C)")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("cross_validation_C.pdf", format="pdf")
plt.show()

matrix_coeff = np.stack(tab_score[:, 3], axis=0)
print(matrix_coeff)
print(matrix_coeff.shape)

ax = plt.subplot(111)
NUM_COLORS = matrix_coeff.shape[1]
LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
NUM_STYLES = len(LINE_STYLES)
cm = plt.get_cmap('Set1')

for i in range(0, matrix_coeff.shape[1]):
    lines = ax.plot(tab_logC, matrix_coeff[:, i])
    lines[0].set_color(cm(i // NUM_STYLES))
    lines[0].set_linestyle(LINE_STYLES[i % NUM_STYLES])

plt.xlabel("log(C)")
plt.xticks(label=np.log(tab_C))

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])
ax.legend(np.arange(0, matrix_coeff.shape[1]), loc='center left', bbox_to_anchor=(1, 0.5))

plt.savefig("cross_validation_C_coeff.pdf", format="pdf")
plt.show()

print("SVM method with RFE")


def extract_feature(clf, X, y):
    pass

# 使用支持向量机（SVM）和递归特征消除（RFE）优化模型
def rfe_svm(X, y):
    clf = linear_model.SGDClassifier(loss='hinge', penalty='elasticnet', max_iter=1000, alpha=1e-9, tol=1e-3,
                                     random_state=123456, class_weight={0: 0.044, 1: 1 - 0.044})
    cv = model_selection.ShuffleSplit(n_splits=10, test_size=0.1, random_state=123456)

    nb_features = X.shape[1]
    print(nb_features)

    scores = model_selection.cross_validate(clf, X, y, cv=cv, scoring=['precision', 'recall', 'f1'],
                                            return_train_score=True)
    print(scores)

    if nb_features > 1:
        rfe = feature_selection.RFE(clf, n_features_to_select=nb_features - 1, step=1)
        rfe.fit(X, y)
        output = rfe_svm(rfe.transform(X), y)

        output.append(
            [nb_features, np.mean(scores['test_precision']), np.mean(scores['test_recall']), np.mean(scores['test_f1']),
             rfe.support_, rfe.ranking_])
        return output
    else:
        return [
            [nb_features, np.mean(scores['test_precision']), np.mean(scores['test_recall']), np.mean(scores['test_f1']),
             [True], [1]]]


results = np.array(rfe_svm(X_train, y_train))
print(results)

plt.plot(results[:, 0], results[:, 1])
plt.plot(results[:, 0], results[:, 2])
plt.plot(results[:, 0], results[:, 3])
plt.legend(["test_precision", "test_recall", "test_f1"])
plt.xlabel("Number of features")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("cross_validation_rfe.pdf", format="pdf")
plt.show()

# 通过交叉验证调整类权重和正则化参数 𝛼
def apply_svm_cross_validation(X, y,
                               svc_args={'loss': 'hinge', 'penalty': 'elasticnet', 'max_iter': 1000, 'alpha': 0.001,
                                         'tol': 1e-3, 'random_state': 123456, 'class_weight': None}):
    clf = linear_model.SGDClassifier(**svc_args)
    cv = model_selection.ShuffleSplit(n_splits=10, test_size=0.1, random_state=123456)
    scores = model_selection.cross_validate(clf, X, y, cv=cv, scoring=['precision', 'recall', 'f1'],
                                            return_train_score=True)
    print(scores)
    return [np.mean(scores['test_precision']), np.mean(scores['test_recall']), np.mean(scores['test_f1'])]


tab_class_weight = np.linspace(0, 0.1, 10)
print(tab_class_weight)

tab_score = np.array([apply_svm_cross_validation(X_train, y_train,
                                                 {'loss': 'hinge', 'penalty': 'elasticnet', 'max_iter': 1000,
                                                  'alpha': 0.001, 'tol': 1e-3, 'random_state': 123456,
                                                  'class_weight': {0: w, 1: 1 - w}}) for w in tab_class_weight])
print(tab_score)

plt.plot(tab_class_weight, tab_score[:, 0])
plt.plot(tab_class_weight, tab_score[:, 1])
plt.plot(tab_class_weight, tab_score[:, 2])
plt.legend(["test_precision", "test_recall", "test_f1"])
plt.xlabel("Botnet class weight")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("cross_validation_class_weight_svm.pdf", format="pdf")
plt.show()

tab_C = np.logspace(-16, -8, 9)
tab_logC = np.log10(tab_C)
print(tab_C)
print(tab_logC)

tab_score = np.array([apply_svm_cross_validation(X_train, y_train,
                                                 {'loss': 'hinge', 'penalty': 'elasticnet', 'max_iter': 1000,
                                                  'alpha': C, 'tol': 1e-3, 'random_state': 123456,
                                                  'class_weight': {0: 0.044, 1: 1 - 0.044}}) for C in tab_C])
print(tab_score)

plt.plot(tab_logC, tab_score[:, 0])
plt.plot(tab_logC, tab_score[:, 1])
plt.plot(tab_logC, tab_score[:, 2])
plt.legend(["test_precision", "test_recall", "test_f1"])
plt.xlabel("log(alpha)")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("cross_validation_alpha.pdf", format="pdf")
plt.show()
