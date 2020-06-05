import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10,452,234,134,224])
y = np.array ([1,2,3,4,5,6,7,8,9,10,432,234,264,513])

x2 = np.array([4,5,6])
#2.모델구성
from keras.models import Sequential
from keras.layers import Dense 
#print ("Hello World")

# model = Sequential()

# model.add(Dense(2, input_dim =1))
# model.add(Dense(3))
# model.add(Dense(4))
# model.add(Dense(5))
# model.add(Dense(4))
# model.add(Dense(3))
# model.add(Dense(2))
# model.add(Dense(1))

from keras.models import load_model
model=load_model('./model/sample/test/check-56-0.0335.hdf5')

model.compile(loss='mse', optimizer = 'adam', metrics=['accuracy'])
# model.fit (x, y, epochs=1000,validation_split=0.2)
model.fit (x, y, epochs=1000,validation_split=0.2)

model.save('./model/model_test01.h5')
#4 평가예측
loss, acc = model.evaluate(x, y)
print("acc:",acc)

y_predict = model.predict(x2)
print(y_predict)
 