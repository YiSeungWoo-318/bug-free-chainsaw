from sklearn.datasets import load_breast_cancer
import tensorflow as tf
import keras
import random
import numpy as np
from sklearn import datasets
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from keras.utils import np_utils




a=pd.read_csv("./data/csv/winequality-white.csv", index_col=None, header=None, encoding='cp949',sep=';')

print(a)
print(a.shape)

a=np.array(a)
Y = a[0, :]
X = a[1:, :]
print(Y)

for i in range(len(X)):
       for j in range(len(X[i])):
          X[i,j]=float(X[i,j])




model = RandomForestRegressor()


model.fit(X, Y)

# y_predict = model.predict(X)
# acc=model.score(X,Y)

# # acc=accuracy_score(Y, y_predict)
# print(" 의 예측 결과 : ", y_predict)
# print("acc = ", acc)
# # print("score:", score)

