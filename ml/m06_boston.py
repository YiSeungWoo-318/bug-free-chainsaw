from sklearn.datasets import load_boston

import tensorflow as tf
import keras
import random
import numpy as np
from sklearn import datasets
import pandas as pd

#분류
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from keras.utils import np_utils
from sklearn import preprocessing


boston = datasets.load_boston()
X = boston.data
Y = boston.target

# std=StandardScaler()
# std.fit(X)
# std.transform(X)

# lab_enc = preprocessing.LabelEncoder()
# Y = lab_enc.fit_transform(Y)

model = RandomForestClassifier()
print(X.shape)
print(Y.shape)


# X=X.reshape(506,13,1)
# Y=Y.reshape(X.shape[0],1)
model.fit(X, Y)

y_predict =model.predict(X)
# Y=Y[:, np.newaxis] 
# Y=Y.reshape(1,506)
# y_predict=y_predict.reshape(13,506)
print(Y.shape)
print(y_predict.shape)
acc=model.score(X, Y)
print(" 의 예측 결과 : ", y_predict)
print("acc = ", acc)


r2=r2_score(Y, y_predict)
print(r2)
