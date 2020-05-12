'''
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([101,102,103,104,105,106,107,108,109,110])
y_test = np.array([101,102,103,104,105,106,107,108,109,110])

model = Sequential()
model.add(Dense(5, input_dim =1, activation ='relu'))
model.add(Dense(3))
model.add(Dense(1, activation ='relu'))

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_data = (x_train, y_train))
loss, acc = model.evaluate(x_test, y_test, batch_size =1)

print("loss :", loss)
print("acc:",  acc)

output = model.predict(x_test)
print("결과물 : \n", output)

#acc이 0 or 1로 수렴 모델 수정 

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([101,102,103,104,105,106,107,108,109,110])
y_test = np.array([101,102,103,104,105,106,107,108,109,110])

model = Sequential()
model.add(Dense(5, input_dim =1))
model.add(Dense(3))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3000, batch_size=1, validation_data = (x_train, y_train))
loss, acc = model.evaluate(x_test, y_test, batch_size =1)

print("loss :", loss)
print("acc:",  acc)

output = model.predict(x_test)
print("결과물 : \n", output)
#model 구성에 activation = 'relu'를 제거함으로서 acc=1.0을 얻는 것을 확인하였다
#model relu는 적합하지 않다.(0,1로 수렴하기 때문에) epochs를 줄임으로서 overfitting된 부분이 감소한 것을 확인하였다.


최적화된 배치사이즈 찾기

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([101,102,103,104,105,106,107,108,109,110])
y_test = np.array([101,102,103,104,105,106,107,108,109,110])

model = Sequential()
model.add(Dense(5, input_dim =1))
model.add(Dense(3))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=300, batch_size=1, validation_data = (x_train, y_train))
loss, acc = model.evaluate(x_test, y_test, batch_size =1)

print("loss :", loss)
print("acc:",  acc)

output = model.predict(x_test)
print("결과물 : \n", output)
epochs = 300 일때 batch사이즈 32는 결과 x batch사이즈 1일때는 결과 정확하므로 1에서 순차적으로 숫자를 증가하기로 하였다. 배치 2여도 acc =0이 된다
따라서 1이 최적이다. 단 오차범위 약 0.02이므로 목표치를 0.002로 설정하고 다시 해보기로 했다. 단, epochs값 300이하로 '''


from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([101,102,103,104,105,106,107,108,109,110])
y_test = np.array([101,102,103,104,105,106,107,108,109,110])

model = Sequential()
model.add(Dense(5, input_dim =1))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=300, batch_size=1, validation_data = (x_train, y_train))
loss, acc = model.evaluate(x_test, y_test, batch_size =1)

print("loss :", loss)
print("acc:",  acc)

output = model.predict(x_test)
print("결과물 : \n", output)
'''오차범위 0.0002의 값을 얻었으므로 만족, 신경망 모양을 고쳤다 ''' 