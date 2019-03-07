# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 00:01:42 2018

@author: Administrator
"""

'''This is machine learning method to compare'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV, MultiTaskLasso
from sklearn.datasets import make_regression
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn import preprocessing

may = np.load('3D-Block-2016-05.npy') 
block = may
# 把Block切片，每一个label对应的training frame是它前1小时，前一天同时段附近，前一周同时段附近，共7个frame
precedent_frames = []
label_frames = []
num = block.shape[0]
gbm = lgb.LGBMRegressor(num_leaves=31)

precedent = np.zeros((7, 64, 64, 2))
label = np.zeros((1, 64, 64, 2))

for i in range(337, num):
    label = block[i, :, :, :]   # label是当前选择的frame
    label = np.reshape(label, (1,64,64,2))
    label_frames.append(label)
    
    precedent[0:2, :, :, :] = block[i-2:i, :, :, :]   # 使用过去的对应时段作为预测的frame, 这是前1小时
    precedent[2:4, :, :, :] = block[i-48:i-46, :, :, :]   # 前一天
    precedent[4:7, :, :, :] = block[i-337:i-334, :, :, :]  # 前一周
    precedent_frames.append(precedent)
    
#regr = (max_depth=8, random_state=0,n_estimators=1000)
model = MultiTaskLasso(alpha=1)  

X_train, X_val, y_train, y_val = train_test_split(precedent_frames, label_frames, test_size=0.2, random_state=4)
# 转化为5D的numpy数组，训练集(920,7,64,64,2)， 测试集(231,1,64,64,2)
X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array(X_val)
y_val = np.array(y_val)

print(X_train.shape)
print(X_val.shape)
print(y_train.shape)
print(y_val.shape)
# 把5D数据转化为randomForest输入的2D数据
X_train = X_train.reshape((920, 7*64*64*2))
X_val = X_val.reshape((231, 7*64*64*2))
y_train = y_train.reshape((920, 1*64*64*2))
y_val = y_val.reshape((231, 1*64*64*2))

model.fit(X_train, y_train)
y_pred = model.predict(X_val)

from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_val, y_pred))