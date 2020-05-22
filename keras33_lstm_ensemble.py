
'''
x의 shape = (batch_size, timesteps, feature)
input_shape = (timesteps, feature)
input_length = timesteps, input_dim = feature
-
'''


import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, LSTM ,Input

#1. 데이터

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]])
x1 = np.array([[10,20,30], [20,30,40], [30,40,50], [40,50,60],[50,60,70],[60,70,80],[70,80,90],[80,90,100],[90,100,110],[100,110,120],[2,3,4],[3,4,5],[4,5,6]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x1_predict = np.array([65,75,85])
x_predict = np.array([55,65,75])


x = x.reshape(x.shape[0],x.shape[1],1)
x1 = x1.reshape(x1.shape[0],x.shape[1],1)

from sklearn.model_selection import train_test_split
x_train, x_test = train_test_split(x, random_state = 1 , train_size=0.75)
x1_train, x1_test = train_test_split(x1, random_state = 2 , train_size=0.75)
y_train, y_test = train_test_split(y, random_state = 3 , train_size=0.75)


input1 = Input(shape=(3,1))
dense1 = LSTM(5)(input1)
dense2 = Dense(5)(dense1)
dense3 = Dense(5)(dense2)
dense4 = Dense(5)(dense3)
dense5 = Dense(5)(dense4)
dense6 = Dense(5)(dense5)

input2 = Input(shape=(3,1))
dense1_1 = LSTM(55)(input2)
dense2_1 = Dense(55)(dense1_1)
dense3_1 = Dense(55)(dense2_1)
dense4_1 = Dense(55)(dense3_1)
dense5_1 = Dense(55)(dense4_1)
dense6_1 = Dense(55)(dense5_1)
from keras.layers.merge import concatenate
merge1=concatenate([dense6,dense6_1])
merge2 = Dense(30)(merge1)

output1 = Dense(1)(merge2)

model = Model(inputs = [input1, input2], outputs=output1)


model.compile(loss = 'mse', optimizer='adam')

from keras.callbacks import EarlyStopping
early_stopping=EarlyStopping(monitor="loss", patience = 300000, mode = 'min')

model.fit([x_train,x1_train],y_train,validation_split=(0.25),epochs=1000000,batch_size = 1,callbacks =[early_stopping])
E=model.evaluate([x_test,x1_test],y_test)

x_predict = x_predict.reshape(1,3,1)  
x1_predict = x1_predict.reshape(1,3,1) 

y_predict = model.predict([x_predict, x1_predict])

print(y_predict)
# from sklearn.metrics import mean_squared_error as mse, r2_score
# def RMSE(v,t):
#     return np.sqrt(mse(v,t))

# RMSE1 = RMSE(y,y_predict)
# r2 = r2_score(y,y_predict)

# print(RMSE1, r2)
