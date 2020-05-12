#1. 데이터

import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array ([1,2,3,4,5,6,7,8,9,10])

x2 = np.array([4,5,6])


#2.모델구성
from keras.models import Sequential
from keras.layers import Dense 
#print ("Hello World")

model = Sequential()

model.add(Dense(2, input_dim =1))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3.훈련
model.compile(loss='mse', optimizer = 'adam', metrics=['accuracy'])
model.fit (x, y, epochs=10000)

#4 평가예측
loss, acc = model.evaluate(x, y)
print("acc:",acc)

y_predict = model.predict(x2)
print(y_predict)
 