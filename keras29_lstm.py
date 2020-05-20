import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense,LSTM,Input

#데이터
x=np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
y=np.array([4,5,6,7])
y2=np.array([[4,5,6,7]])
y3=np.array([[4],[5],[6],[7]])


#x = x.reshape(4,3,1)
x = x.reshape(x.shape[0], x.shape[1], 1)
#4,3,1
y = y.reshape(1,y.shape[0],1)

#4,1
y2 = y2.reshape(1,4,1)
# #4,1
y3 = y3.reshape(1,4,1) 
#1,4,1
# print(x.shape)
# print(y.shape)
# print(y2.shape)
# print(y3.shape)

input1 = Input(shape=(3,1))
dense1 = (Dense(5))(input1)
dense2 = (Dense(5))(dense1)
dense3 = (Dense(5))(dense2)
dense4 = (Dense(5))(dense3)

#----------------------------------------------------------------

output1 = (Dense(5))(dense4)
output2 = (Dense(5))(output1)
output3 = (Dense(1))(output2)


output1_1 = (Dense(5))(dense4)
output2_1 = (Dense(5))(output1_1)
output3_1 = (Dense(1))(output2_1)



output1_2 = (Dense(5))(dense4)
output2_2 = (Dense(5))(output1_2)
output3_2 = (Dense(1))(output2_2)



model = Model(inputs = input1, outputs = [output3, output3_1, output3_2])



model.summary()



# #3. 훈련
# model.compile(loss = 'mse', optimizer='adam', metrics=['mse'])
# model.fit([x1_train, x2_train], [y1_train,y2_train], epochs=100, batch_size=1, validation_split=(0.2), verbose = 1)
# # mse 는 회귀 acc는 분류 회귀는 1차함수 분류는 예측값의 범위가 정해져 있다.


# #4. 평가, 예측
# E = model.evaluate([x1_test, x2_test], [y1_test, y2_test] , batch_size=1) #x,y를 평가하여 loss와 acc에 반환하겠다.


# #y_pred = model.predict(x_pred)
# #print("y_predict : ", y_pred)
# print([x1_test, x2_test])
# y1_predict, y2_predict = model.predict([x1_test, x2_test])
# print(y1_predict)
# print(y2_predict)

 
# # RMSE 구하기
# from sklearn.metrics import mean_squared_error
# def RMSE(y_test, y_predict ):
#     return np.sqrt(mean_squared_error(y_test, y_predict))  

# RMSE1 = RMSE(y1_test, y1_predict)

# RMSE2 = RMSE(y2_test, y2_predict)

# print("Rmse1 : ", RMSE1)
# print("Rmse2 : ", RMSE2)
# print("Rmse : ", (RMSE1+RMSE2)/2)



# # R2 구하기
# from sklearn.metrics import r2_score
# r2_1y_predict = r2_score(y1_test, y1_predict)
# r2_2y_predict = r2_score(y2_test, y2_predict)

# print("R2_1: ", r2_1y_predict)
# print("R2_2: ", r2_2y_predict)
# print("R2: ", (r2_1y_predict+r2_2y_predict)/2)
# # 1. R2 0.5 이하








# ''''
# model = Sequential()

# model.add(LSTM(10, activation = 'relu', input_shape =(3,1)))
# model.add(Dense(5))
# model.add(Dense(1))

# model.summary()


# model.compile(optimizer = 'adam', loss='mse')
# model.fit(x,y,epochs = 100)

# x_input = np.array([5,6,7])
# x_input = x_input.reshape(1,3,1)
# print(x_input)
# yhat = model.predict(x_input)
# print(yhat)

# '''