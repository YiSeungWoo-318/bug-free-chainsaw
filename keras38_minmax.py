
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

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],[2000,3000,4000],[3000,4000,5000],[4000,5000,6000],[100,200,300]])

y = np.array([4,5,6,7,8,9,10,11,12,13,5000,6000,7000,400])


x_predict = np.array([55,65,75])

x_predict= x_predict.reshape(1,x_predict.shape[0])






from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
scaler=StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

x_predict = scaler.transform(x_predict)


print(x)
print(x_predict)

x = x.reshape(x.shape[0], x.shape[1], 1)

input1 = Input(shape=(3,1))
dense1 = LSTM(10,return_sequences=True)(input1)
dense2 = LSTM(5)(dense1)
dense3 = Dense(4)(dense2)
dense4 = Dense(2)(dense3)
output1 = Dense(1)(dense4)

model = Model(inputs = input1, outputs=output1)

#Dense layer는 2차원만 입력받는다. lstm은 3차원을 받고 있다. 따라서 오류바생
#3차원으로 바꿔 주는 기능을 리턴 시퀀스라고 한다. return_sequences=True

model.summary()








model.compile(loss = 'mse', optimizer='adam')

from keras.callbacks import EarlyStopping
early_stopping=EarlyStopping(monitor="loss", patience = 300, mode = 'min')

model.fit(x,y,epochs=1000,batch_size = 32,callbacks =[early_stopping],validation_split=(0.25))

E=model.evaluate(x,y)
x_predict = x_predict.reshape(1,3,1)

y_predict = model.predict(x_predict)

print(y_predict)


# # from sklearn.metrics import mean_squared_error as mse, r2_score
# # def RMSE(v,t):
# #     return np.sqrt(mse(v,t))

# # RMSE1 = RMSE(y,y_predict)
# # r2 = r2_score(y,y_predict)

# # print(RMSE1, r2)
