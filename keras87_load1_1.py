

import numpy as np

import matplotlib.pyplot as plt



from keras.datasets import mnist



(x_train, y_train),(x_test, y_test) = mnist.load_data()



print(x_train[0])

print('y_train :',y_train[0])





#plt.imshow(x_train[1], 'gray')

#plt.show()



#from sklearn.preprocessing import OneHotEncoder

#hot = OneHotEncoder()



## OneHotEncoding

from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train)

y_test = np_utils.to_categorical(y_test)



## 정규화

#from sklearn.preprocessing import MinMaxScaler

x_train = x_train.reshape(60000,28,28,1).astype('float32') / 255 ## float 안해줘도 되지 않나?

x_test = x_test.reshape(10000,28,28,1).astype('float32') / 255







from keras.models import load_model
model=load_model('./model/model_test01.h5')


from keras.models import Model

from keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout, Input

dense10= Dense(10, name='f') 

model.summary()
'''

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])



model.fit(x_train, y_train, batch_size = 500, epochs=60, validation_split=0.3)


loss, acc = model.evaluate(x_test,y_test)


print('loss :',loss)

print('acc :',acc)

# loss : 0.04145813554450724
# acc : 0.9921000003814697
'''