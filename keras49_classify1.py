import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.datasets import reuters
import tensorflow as tf
from numpy import argmax
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils.np_utils import to_categorical
x = np.array(range(1,11))
y= np.array([1,2,3,4,5,1,2,3,4,5])
min_y = np.min(y)

# y= np_utils.to_categorical(y=y, num_classes=10)
# y = np_utils.to_categorical(y,6)
y = np_utils.to_categorical(y)
# x = np_utils.to_categorical(x)
# y=y.reshape(y.shape[0],y.shape[1],1)
y=y[:,1:6]

from keras.models import Sequential, Model
from keras.layers import Dense,Input,LSTM

model=Sequential()
model.add(Dense(15,activation='relu', input_dim =1))
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(5,activation='relu'))
model.add(Dense(len(y[0]),activation='softmax'))


#
model.compile(loss='categorical_crossentropy', optimizer= 'adam', metrics = ['acc'])
model.fit(x,y,epochs=5000)
loss,acc=model.evaluate(x,y)




y1 = model.predict(x)

print(len(y1))

# matrix = confusion_matrix(y, y1)
print(y1)
y_predict2 = []
# def p1():
#     return y1
for i in range(len(y1)):
    
    y_predict2.append(np.argmax(y1[i]))

y_predict2 += min_y

# print(np.argmax(y_predict2,axis=1))
print(y_predict2)
print(np.around(loss, 5))
from sklearn.metrics import mean_squared_error as mse

def RMSE(y_test,y_predict):
    return np.sqrt(mse(y_test,y_predict))

print("RMSE:",RMSE(y,y1))

from sklearn.metrics import r2_score

r2=r2_score(y,y1)

print("R2:",r2)
