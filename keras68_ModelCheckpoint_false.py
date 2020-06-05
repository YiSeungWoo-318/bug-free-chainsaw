import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.datasets import hist
(x_train, y_train),(x_test,y_test)=mnist.load_data()

print(x_train[0])
print('y_train :', y_train[0])

print(x_train.shape)
print(x_test.shape)

print(y_train.shape)
print(y_test.shape)
print(x_train[0].shape)
# plt.imshow(x_train[0],'gray')
# plt.show()

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

model = Sequential()
model.add(Conv2D(10,(2,2),input_shape=(28,28,1)))#10=fileter, ((2,2)=kernel size,  kernel size=2) height,width,channel 행가로세로 색깔
model.add(Conv2D(7,(2,2)))
model.add(Conv2D(5,(2,2),padding='same'))
model.add(Conv2D(5,(2,2)))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(10))
model.summary()

from keras.callbacks import Early_stopping,ModelCheckpoint
modelpath = './model/{epoch:02d}-{val_loss:.4f).hdf5'
checkpoint = ModelCheckpoint(filepath='modelpath', monitor='val_loss',save_best_only=True )

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs=10,validation_split=(0.3),callbacks=checkpoint)
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=32)

print('')
print('loss_and_metrics : ' + str(loss_and_metrics))

from keras.datasets import hist
#강의시간 작성내용
loss_acc = model.evaluate(x_test, y_test)

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc=hist.history['accuracy']
val_acc=hist.history['val_accuracy']


print('acc: ', acc)
print('val_acc : ' val_acc)
print('loss_acc :' loss_acc)






import matplotlib.pyplot as plt
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
plt.legen(loc='upper right')
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






# # 6. 모델 사용하기
# xhat_idx = np.random.choice(x_test.shape[0], 5)


# xhat = x_test[xhat_idx]


# yhat = model.predict_classes(xhat)
# print(yhat)
# for i in range(5):
#     print('True : ' + str(np.argmax(y_test[xhat_idx[i]])) + ', Predict : ' + str(yhat[i]))
    
# #PARAM
# #Total PARAM
