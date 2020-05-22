
'''
x의 shape = (batch_size, timesteps, feature)
input_shape = (timesteps, feature)
input_length = timesteps, input_dim = feature
-
'''


import numpy as np

from keras.models import Sequential
from keras.layers import Dense, GRU

#1. 데이터

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]])

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])


x_predict = np.array([50,60,70])

x = x.reshape(x.shape[0],x.shape[1],1)



model = Sequential()


model.add(GRU(20, activation='relu', input_shape=(3,1))) 
model.add(Dense(49,activation='relu'))
model.add(Dense(254,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(14,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(1))

model.summary()


model.compile(loss = 'mse', optimizer='adam')

from keras.callbacks import EarlyStopping
early_stopping=EarlyStopping(monitor="loss", patience = 2000, mode = 'min')

model.fit(x,y,epochs=3000,batch_size = 2,callbacks =[early_stopping])
E=model.evaluate(x,y)
print(E)

x_predict = x_predict.reshape(1,3,1)  

y_predict = model.predict(x_predict)

print(y_predict)


# from sklearn.metrics import mean_squared_error as mse, r2_score
# def RMSE(v,t):
#     return np.sqrt(mse(v,t))

# RMSE1 = RMSE(y,y_predict)
# r2 = r2_score(y,y_predict)

# print(RMSE1, r2)