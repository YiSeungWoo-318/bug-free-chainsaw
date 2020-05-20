# #1.데이터
# import numpy as np 
# x_train=np.array([1,2,3,4,5,6,7,8,9,10])
# y_train=np.array([1,2,3,4,5,6,7,8,9,10])
# x_test=np.array([11,12,13,14,15])
# y_test=np.array([11,12,13,14,15])
# x_pred=np.array([16,17,18])
# #predict

# #2.모델구성
# from keras.models import Sequential
# from keras.layers import Dense
# model_youngsun = Sequential()

# model_youngsun.add(Dense(5, input_dim=1))
# model_youngsun.add(Dense(5))
# model_youngsun.add(Dense(5))
# model_youngsun.add(Dense(5))
# model_youngsun.add(Dense(1)) 


# #3.훈련
# model_youngsun.compile(loss='mse', optimizer='adam', metrics=['mse'])
# model_youngsun.fit(x_train,y_train,epochs=1000, batch_size=1)

# #4.평가, 예측 
# loss, mse = model_youngsun.evaluate(x_test,y_test)
# print("loss:",loss)
# print("mse:",mse)

# y_pred=model_youngsun.predict(x_pred)
# print("y_predict:", y_pred)
print("pwd")
import pandas as pd
adult = pd.read_csv('adult.csv')