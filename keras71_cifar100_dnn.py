from keras.datasets import cifar100
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D
from keras.layers import Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test,y_test)=cifar100.load_data()

# print(x_train.max)

# print(x_train[0])
# print('y_train[0] : ', y_train[0])
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)
# plt.imshow(x_train[0])
# plt.show()
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)
x_train = x_train/255
#minmax를 쓰지 않고 최대값으로 나누어 일로 만들어준다. 
#  x-최소
# 최대-최소
x_train = x_train.reshape(50000,32*32*3).astype('float32')/255
x_test = x_test.reshape(10000,32*32*3).astype('float32')/255

model = Sequential()
model.add(Dense(100,input_shape=(32*32*3,),activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='elu'))
model.add(Dense(50,activation='elu'))
model.add(Dense(100,activation='softmax'))
model.summary
from keras.callbacks import EarlyStopping,ModelCheckpoint

from keras.callbacks import ModelCheckpoint
modelpath = './model/sample/cifar100/check-{epoch:02d}-{val_loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath=modelpath,monitor='val_loss', save_best_only=True, save_weights_only=False, verbose=1)
earlystopping=EarlyStopping(monitor='loss', patience=100, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['acc'])
hist=model.fit(x_train, y_train, epochs=10, callbacks=[checkpoint, earlystopping],validation_split=(0.3))

model.save('./model/sample/cifar100/model_test01.h5')
model.save_weights('./model/sample//cifar100/test_weight1.h5')
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=14)
#------------------------------------------------------------------------------------------------------------------#
plt.figure(figsize=(10,6))
plt.subplot(2,1,1,)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss') #plt.plot(x,y,hist.history['loss'])- x, y 별도추가 추가하지 않으면 epochs 순으로 기록 

plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
# plt.plot(hist.history['acc'])
# plt.plot(hist.history['val_acc'])
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
# plt.legend(['loss', 'val_loss'])
plt.legend(loc='upper right')
plt.show()

plt.subplot(2,1,2,)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
# plt.plot(hist.history['acc'])
# plt.plot(hist.history['val_acc'])
plt.grid()
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['loss', 'val_acc'])
plt.show()
#------------------------------------------------------------------------------------------------------------------#
print('')
print('loss_and_metrics : ' + str(loss_and_metrics))

# 6. 모델 사용하기
xhat_idx = np.random.choice(x_test.shape[0], 5)
xhat = x_test[xhat_idx]
yhat = model.predict_classes(xhat)
print(yhat)
for i in range(5):
    print('True : ' + str(np.argmax(y_test[xhat_idx[i]])) + ', Predict : ' + str(yhat[i]))