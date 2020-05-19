
# x1, y1을 훈련시켜 x2 varible 주어졌을 때 Y-Predict 값을 예측하시오
import numpy as np

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array ([1,2,3,4,5,6,7,8,9,10])

x2 = np.array([4,5,6])



from keras.models import Sequential 
from keras.layers import Dense

model = Sequential()
model.add(Dense(1, input_dim = 1))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(1))

model.compile(loss= 'mse', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x,y, epochs = 100, batch_size = 1)

loss, acc = model.evaluate(x,y)

Y_predict = model.predict(x2)

print("Y_predict : ", Y_predict)




