#1. 데이터
import numpy as np
x1 = np.array([range(1, 101), range(311,411),range(100)])
y1=np.array([range(711, 811), range(711,811),range(100)])

x2=np.array([range(101, 201), range(411,511),range(100,200)])
y2=np.array([range(501, 601), range(711,811),range(100)])



x1 = np.transpose(x1)
y1 = np.transpose(y1)

x2 = np.transpose(x2)
y2 = np.transpose(y2)


from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, train_size=0.75)

from sklearn.model_selection import train_test_split
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, train_size=0.75)

#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input
#--------------------------------------#

input1 = Input(shape=(3,))

dense1_1 = Dense(5, activation = 'relu',name = 'dense1_1')(input1)
dense1_2 = Dense(5, activation = 'relu',name = 'dense1_2')(dense1_1)
dense1_3 = Dense(5, activation = 'relu',name = 'dense1_3')(dense1_2)
dense1_4 = Dense(5, activation = 'relu',name = 'dense1_4')(dense1_3)
dense1_5 = Dense(3, activation = 'relu',name = 'dense1_5')(dense1_4)



#-------------------------------------#


input2 = Input(shape=(3,))

dense2_1 = Dense(5, activation = 'relu',name = 'dense2_1')(input2)
dense2_2 = Dense(6, activation = 'relu',name = 'dense2_2')(dense2_1)
dense2_3 = Dense(10, activation = 'relu',name = 'dense2_3')(dense2_2)
dense2_4 = Dense(6, activation = 'relu',name = 'dense2_4')(dense2_3)
dense2_5 = Dense(3, activation = 'relu',name = 'dense2_5')(dense2_4)




#-------------------------------------#

from keras.layers.merge import concatenate
merge1=concatenate([dense1_5, dense2_5])

middle1 = Dense(3)(merge1)
middle1 = Dense(10)(middle1)
middle1 = Dense(3)(middle1)

## output 모델구성

output1 = Dense(3)(middle1)
output1_2 = Dense(5)(output1)
output1_3 = Dense(5)(output1_2)
output1_4 = Dense(5)(output1_3)
output1_5 = Dense(3)(output1_4)

output2 = Dense(3) (middle1)
output2_2 = Dense(10)(output2)
output2_3 = Dense(5)(output2_2)
output2_4 = Dense(5)(output2_3)
output2_5= Dense(3)(output2_4)

model= Model(inputs=[input1, input2], outputs = [output1_5,output2_5])


model.summary()



#3. 훈련
model.compile(loss = 'mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train, x2_train], [y1_train,y2_train], epochs=100, batch_size=1, validation_split=(0.2), verbose = 1)
# mse 는 회귀 acc는 분류 회귀는 1차함수 분류는 예측값의 범위가 정해져 있다.


#4. 평가, 예측
a,b, c, x, y = model.evaluate([x1_test, x2_test], [y1_test, y2_test] , batch_size=1) #x,y를 평가하여 loss와 acc에 반환하겠다.


print("x : ", x)
print("y : ", y)
print("c : ", c)
print("a : ", a)
print("b : ", b)


#y_pred = model.predict(x_pred)
#print("y_predict : ", y_pred)
print([x1_test, x2_test])
y1_predict, y2_predict = model.predict([x1_test, x2_test])
print(y1_predict)
print(y2_predict)

 
# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict ):
    return np.sqrt(mean_squared_error(y_test, y_predict))  

RMSE1 = RMSE(y1_test, y1_predict)

RMSE2 = RMSE(y2_test, y2_predict)

print("Rmse1 : ", RMSE1)
print("Rmse2 : ", RMSE2)
print("Rmse : ", (RMSE1+RMSE2)/2)



# R2 구하기
from sklearn.metrics import r2_score
r2_1y_predict = r2_score(y1_test, y1_predict)
r2_2y_predict = r2_score(y2_test, y2_predict)

print("R2_1: ", r2_1y_predict)
print("R2_2: ", r2_2y_predict)
print("R2: ", (r2_1y_predict+r2_2y_predict)/2)
# 1. R2 0.5 이하
# 2. layers는 5개이상
# 3. 노드의 갯수 10개이상
# 4. batch_size 8이하
# 5. epochs는 30이상 
