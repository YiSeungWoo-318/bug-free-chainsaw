
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

x_predict = x_predict.reshape(3,1)

# print(x.shape) #13,3,None
# print(y.shape) #13,1,none
# print(x_predict.shape) #3,1,none

x_predict = np.transpose(x_predict)
print(x_predict.shape)



input1 = Input(shape=(3,))
dense1 = Dense(3)(input1)
dense2 = Dense(3)(dense1)
dense3 = Dense(3)(dense2)
dense4 = Dense(3)(dense3)
dense5 = Dense(3)(dense4)
dense6 = Dense(3)(dense5)
output1 = Dense(1)(dense6)

model = Model(inputs = input1, outputs=output1)

#Dense layer는 2차원만 입력받는다. lstm은 3차원을 받고 있다. 따라서 오류바생
#3차원으로 바꿔 주는 기능을 리턴 시퀀스라고 한다. return_sequences=True

model.summary()






model.compile(loss = 'mse', optimizer='adam')



model.fit(x,y,epochs=3000,batch_size = 1)

E=model.evaluate(x,y)

y_predict = model.predict(x_predict)

print(y_predict.shape)

print(y_predict)






# from sklearn.metrics import mean_squared_error as mse, r2_score
# def RMSE(v,t):
#     return np.sqrt(mse(v,t))

# RMSE1 = RMSE(y,y_predict)
# r2 = r2_score(y,y_predict)

# print(RMSE1, r2)
