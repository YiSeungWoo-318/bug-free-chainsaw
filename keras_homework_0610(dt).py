import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

from keras.models import Sequential, Model

from keras.layers import Dense, Dropout, Input

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from keras.wrappers.scikit_learn import KerasRegressor

from keras.losses import MeanAbsoluteError

from keras.layers import LeakyReLU
from keras.models import Sequential, Model

from keras.layers import Dense, Dropout, Input

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

from keras.callbacks import EarlyStopping

from sklearn.decomposition import PCA

from keras.layers.merge import concatenate  
leaky = LeakyReLU(alpha = 0.2)
#1. data
train = pd.read_csv('./data/csv/comp/train.csv', index_col= 0 , header = 0, sep=',')
test = pd.read_csv('./data/csv/comp/test.csv', index_col= None , header = 0, sep=',')
submission = pd.read_csv('./data/csv/comp/sample_submission.csv', index_col= 0 , header = 0, sep=',')



print('train.shape: ', train.shape)              # (10000, 75)  = x_train, test
print('test.shape: ', test.shape)                # (10000, 71)  = x_predict
print('submission.shape: ', submission.shape)    # (10000, 4)   = y_predict
print(train.isnull().sum())                      # train에 있는 null값의 합

# rho           0
# 650_src       0
# 660_src       0
# 670_src       0
# 680_src       0
#            ...
# 990_dst    1987
# hhb           0
# hbo2          0
# ca            0
# na            0
# Length: 75, dtype: int64



train = train.interpolate()                       # 보간법 : 선형보간 / 모델을 돌려서 예측 값을 넣음 / 맨 앞행은 안 채워짐

print(train.isnull().sum())                       #        : 구간을 잘라서 선에 맞게 빈자리를 채워줌

# rho        0                                    # column별 보관 : 옆의 column에 영향 X

# 650_src    0

# 660_src    0

# 670_src    0

# 680_src    0

#           ..

# 990_dst    0

# hhb        0

# hbo2       0

# ca         0

# na         0

# Length: 75, dtype: int64



test = test.interpolate()



x = train.iloc[:, :-4]                           

y = train.iloc[:, -4:]

x = x.fillna(method = 'bfill')
test = test.fillna(method = 'bfill')

# print(x.info())

# print(test.info())



x = x.values

y = y.values

x=x*10000
y=y*10000
from sklearn.tree import DecisionTreeClassifier
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.2, random_state=42)

print(x_train.shape) #113,30
print(x_test.shape) #456,30
print(y_train.shape) #113,
print(y_test.shape) #456,


model=DecisionTreeClassifier(max_depth=4)
model.fit(x_train,y_train)
acc = model.score(x_test,y_test)
print(model.feature_importances_)