#1. 데이터

import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array ([1,2,3,4,5,6,7,8,9,10])


#2.모델구성
from keras.models import Sequential
from keras.layers import Dense 
#print ("Hello World")

model = Sequential()

model.add(Dense(1, input_dim =1, activation='relu'))
model.add(Dense(3))
model.add(Dense(5))
model.add(Dense(2))
model.add(Dense(1))
from keras.models import load_model
model=save_model('./model/sample/test/check-56-0.0335.hdf5')
#3.훈련
model.compile(loss='mse', optimizer = 'adam', metrics=['accuracy'])
model.fit (x, y, epochs=1000, batch_size=1)

#4 평가예측
loss, acc = model.evaluate(x, y)
print("acc:",acc)

