import numpy as np
x = np.array([range(1, 101), range(301,401),range(100)])
x = np.transpose(x)
y = np.array([range(601,701),range(601,701),range(601,701)])
y = np.transpose(y)



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75)

#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

input1 = Input(shape=(3,))

dense1_1 = Dense(5, activation = 'relu',name = 'dense1_1')(input1)
dense1_2 = Dense(5, activation = 'relu',name = 'dense1_2')(dense1_1)
dense1_3 = Dense(5, activation = 'relu',name = 'dense1_3')(dense1_2)
dense1_4 = Dense(5, activation = 'relu',name = 'dense1_4')(dense1_3)
dense1_5 = Dense(3, activation = 'relu',name = 'dense1_5')(dense1_4)

output3 = Dense(3) (dense1_5)
output3_2 = Dense(10)(output3)
output3_3 = Dense(5)(output3_2)
output3_4 = Dense(5)(output3_3)
output3_5= Dense(3)(output3_4)

model= Model(inputs=[input1], outputs = [output3_5])

#3. 훈련
model.compile(loss = 'mse', optimizer='adam', metrics=['mse'])

from keras.callbacks import EarlyStopping
early_stopping=EarlyStopping(monitor="loss", patience = 100, mode = 'auto')
model.fit([x_train],[y_train], epochs=500, batch_size=2, validation_split=(0.2), verbose = 1, callbacks =[early_stopping])
# mse 는 회귀 acc는 분류 회귀는 1차함수 분류는 예측값의 범위가 정해져 있다.


#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=1) #x,y를 평가하여 loss와 acc에 반환하겠다.

print("loss : ", loss)


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


#----------------------------------------------------------------------------------------------------------------------------
