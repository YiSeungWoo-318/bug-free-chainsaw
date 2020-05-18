#1. 데이터
import numpy as np
x = np.array([range(1, 101), range(301,401),range(100)])
x = np.transpose(x)
y = np.array(range(701,801))
y = np.transpose(y)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75)

#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

# model = Sequential()
# model.add(Dense(5, input_dim = 3))
# model.add(Dense(4))
# model.add(Dense(3))
# model.add(Dense(1))

input1 = Input(shape=(3,))

dense1 = Dense(5, activation = 'relu')(input1)
dense2 = Dense(5, activation = 'relu')(dense1)
dense3 = Dense(5, activation = 'relu')(dense2)
dense4 = Dense(5, activation = 'relu')(dense3)
dense5 = Dense(5, activation = 'relu')(dense4)
output1 = Dense(1)(dense5)

model = Model(inputs=input1, outputs=output1)


model.summary()


#3. 훈련
model.compile(loss = 'mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=(0.1))
# mse 는 회귀 acc는 분류 회귀는 1차함수 분류는 예측값의 범위가 정해져 있다.


#4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1) #x,y를 평가하여 loss와 acc에 반환하겠다.

print("loss : ", loss)
print("mse : ", mse)

#y_pred = model.predict(x_pred)
#print("y_predict : ", y_pred)
print(x_test)
y_predict = model.predict(x_test)
print(y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict ):
    return np.sqrt(mean_squared_error(y_test, y_predict))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              

print("RMSE: ", RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)

print("R2: ", r2_y_predict)


# 1. R2 0.5 이하
# 2. layers는 5개이상
# 3. 노드의 갯수 10개이상
# 4. batch_size 8이하
# 5. epochs는 30이상 
