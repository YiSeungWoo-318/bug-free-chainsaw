from sklearn.datasets import load_iris

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
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from keras.utils import np_utils

iris = datasets.load_iris()
X = iris.data
Y = iris.target

Y = np_utils.to_categorical(Y)

model = RandomForestClassifier()



model.fit(X, Y)

y_predict = model.predict(X)

score = model.score(X,Y)

acc=accuracy_score(Y, y_predict)
print(" 의 예측 결과 : ", y_predict)
print("acc = ", acc)
print("score은:", score)