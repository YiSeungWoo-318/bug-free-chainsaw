
'''
x의 shape = (batch_size, timesteps, feature)
input_shape = (timesteps, feature)
input_length = timesteps, input_dim = feature
-
'''


import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input

#1. 데이터

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]])

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])


x_predict = np.array([55,65,75])
print(x.shape)
x = x.reshape(x.shape[0],x.shape[1],1)

print(x.shape)

# from sklearn.model_selection import train_test_split
# x_train, x_test = train_test_split(x, random_state = 1 , train_size=0.9)
# y_train, y_test = train_test_split(y, random_state = 3 , train_size=0.9)

# input1 = Input(shape=(3,1))
# dense1 = LSTM(10,return_sequences=True)(input1)
# dense2 = LSTM(10,return_sequences=True)(dense1)
# dense3 = LSTM(10,return_sequences=True)(dense2)
# dense4 = LSTM(10,return_sequences=True)(dense3)
# dense5 = LSTM(10)(dense4)
# dense6 = Dense(60)(dense5)
# dense7 = Dense(20)(dense6)
# dense8 = Dense(4)(dense7)
# output1 = Dense(1)(dense8)

# model = Model(inputs = input1, outputs=output1)

# #Dense layer는 2차원만 입력받는다. lstm은 3차원을 받고 있다. 따라서 오류바생
# #3차원으로 바꿔 주는 기능을 리턴 시퀀스라고 한다. return_sequences=True

# model.summary()








# model.compile(loss = 'mse', optimizer='adam')

# from keras.callbacks import EarlyStopping
# early_stopping=EarlyStopping(monitor="loss", patience = 50000, mode = 'min')

# model.fit(x_train,y_train,epochs=100000,batch_size = 2,callbacks =[early_stopping],validation_split=(0.25))

# E=model.evaluate(x_test,y_test)


# x_predict = x_predict.reshape(1,3,1)  

# y_predict = model.predict(x_predict)

# print(y_predict)


# from sklearn.metrics import mean_squared_error as mse, r2_score
# def RMSE(v,t):
#     return np.sqrt(mse(v,t))

# RMSE1 = RMSE(y,y_predict)
# r2 = r2_score(y,y_predict)

# print(RMSE1, r2)
