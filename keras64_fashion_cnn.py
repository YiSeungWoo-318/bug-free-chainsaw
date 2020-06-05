import numpy as np 
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D
from keras.layers import Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt

#데이터 부르기 
(x_train, y_train), (x_test,y_test)=fashion_mnist.load_data()

#데이터 확인
print(x_train[0])
print('y_train[0] : ', y_train[0])
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
plt.imshow(x_train[0])
plt.show()

#데이터 전처러
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)

x_train = x_train/255
#minmax를 쓰지 않고 최대값으로 나누어 일로 만들어준다. 
#  x-최소
# 최대-최소

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255

#255,255.0,255.
#float32 => 1~255 1.0 ~ 255.0 실수값을로 추정 

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.layers import Dropout
from keras.callbacks import EarlyStopping

model = Sequential()
model.add(Conv2D(100,(2,2),input_shape=(28,28,1)))#10=fileter, ((2,2)=kernel size,  kernel size=2) height,width,channel 행가로세로 색깔
# model.add(Dropout(0.5))
# model.add(Conv2D(10,(2,2)))
model.add(Dropout(0.2))
# model.add(Conv2D(10,(2,2)))
model.add(Conv2D(10,(1,1)))
model.add(MaxPooling2D(pool_size=1))
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='elu'))
model.add(Dense(100,activation='elu'))
model.add(Dense(50,activation='elu'))
model.add(Dense(10,activation='softmax'))
model.summary()


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs=200)
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=14)

print('')
print('loss_and_metrics : ' + str(loss_and_metrics))

# 6. 모델 사용하기
xhat_idx = np.random.choice(x_test.shape[0], 5)


xhat = x_test[xhat_idx]


yhat = model.predict_classes(xhat)
print(yhat)
for i in range(5):
    print('True : ' + str(np.argmax(y_test[xhat_idx[i]])) + ', Predict : ' + str(yhat[i]))
    
#PARAM
#Total PARAM
#acc와 loss결과 설명하시오