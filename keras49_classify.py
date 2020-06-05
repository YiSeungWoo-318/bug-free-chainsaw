import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.datasets import reuters

x = np.array(range(1,11))
y= np.array([1,2,3,4,5,1,2,3,4,5])


# y= np_utils.to_categorical(y=y, num_classes=10)
y = np_utils.to_categorical(y,11)
x = np_utils.to_categorical(x)



# y=y.reshape(y.shape[0],y.shape[1],1)
# x=x.reshape(x.shape[0],x.shape[1],1)
# y=y.reshape(y.shape[0],y.shape[1],1)
# #모델
print(y)
print(x.shape)
print(y.shape)

# from keras.models import Sequential, Model
# from keras.layers import Dense,Input,LSTM

# input1 = Input(shape=(11,1))
# dense1 = LSTM(10,activation='relu',return_sequences=True)(input1)
# # dense2 = Dense(5)(dense1)
# # dense3 = Dense(4)(dense2)
# # dense4 = Dense(2)(dense3)
# output1 = LSTM(1,activation='softmax',return_sequences=True)(dense1)

# model = Model(inputs = input1, outputs=output1)

# # model.add(Dense(15,activation='relu', input_dim =1))
# # # model.add(Dense(100,activation='relu'))
# # # model.add(Dense(100,activation='relu'))
# # # model.add(Dense(100,activation='relu'))
# # # model.add(Dense(50,activation='relu'))
# # # model.add(Dense(5,activation='relu'))
# # model.add(Dense(1,activation='softmax'))



# #
# model.compile(loss='sparse_categorical_crossentropy', optimizer= 'sgd', metrics = ['acc'])
# model.fit(x,y,epochs=30000)

# y1=model.predict(x)
# print(y1)

# from sklearn.metrics import mean_squared_error as mse

# def RMSE(y_test,y_predict):
#     return np.sqrt(mse(y_test,y_predict))

# print("RMSE:",RMSE(y,y1))

# from sklearn.metrics import r2_score

# r2=r2_score(y,y1)

# print("R2:",r2)
