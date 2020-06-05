import numpy as np
import pandas as pd 
iris= np.load('./data/iris.npy')

x=iris[0:150, 0:4]

y=iris[0:150, 4:]
# print(x.shape)

print(list(y).count(0))


from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
Y_1hot = enc.fit_transform(y.reshape(-1, 1)).toarray()
print(y[0], " -- one hot enocding --> ", Y_1hot[0])
print(y[50], " -- one hot enocding --> ", Y_1hot[50])
print(y[100], " -- one hot enocding --> ", Y_1hot[100])

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test =train_test_split(x, Y_1hot, shuffle=True, train_size=0.75)



# df = pd.DataFrame(iris.data, columns=iris.feature_names)
# df.head()

#    sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
# 0                5.1               3.5                1.4               0.2
# 1                4.9               3.0                1.4               0.2
# 2                4.7               3.2                1.3               0.2
# 3                4.6               3.1                1.5               0.2
# 4                5.0               3.6                1.4               0.2

# df.describe()
#        sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
# count         150.000000        150.000000         150.000000        150.000000
# mean            5.843333          3.057333           3.758000          1.199333
# std             0.828066          0.435866           1.765298          0.762238
# min             4.300000          2.000000           1.000000          0.100000
# 25%             5.100000          2.800000           1.600000          0.300000
# 50%             5.800000          3.000000           4.350000          1.300000
# 75%             6.400000          3.300000           5.100000          1.800000
# max             7.900000          4.400000           6.900000          2.500000

# df.info()
#  #   Column             Non-Null Count  Dtype
# ---  ------             --------------  -----
#  0   sepal length (cm)  150 non-null    float64
#  1   sepal width (cm)   150 non-null    float64
#  2   petal length (cm)  150 non-null    float64
#  3   petal width (cm)   150 non-null    float64
# dtypes: float64(4)
# memory usage: 4.8 KB
# None
# 총 4개의 colums 아이리스의 꽃받침과 꽃잎 두께와 길이 분석하여 해당 꽃의 분류 모델이다.
# 'ㅋㅋ


from keras.backend import tensorflow_backend as K

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     tf.config.experimental.set_memory_growth(gpus[0], True)
#   except RuntimeError as e:
#     # gpu메모리 적게 준다.
#     print(e)

from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(4, input_dim=4, activation='selu'))
model.add(Dense(4, activation='elu'))
model.add(Dense(3, activation='softmax'))

import numpy as np
from keras.callbacks import ModelCheckpoint
modelpath = './model/sample/iris/check-{epoch:02d}-{val_loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath=modelpath,monitor='val_loss', save_best_only=True, save_weights_only=False, verbose=1)


from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss',patience=90)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=400, batch_size=2,callbacks=[es,checkpoint],validation_split=0.2)
model.evaluate(X_test, Y_test)


model.save('./model/sample/iris/model_test01.h5')
model.save_weights('./model/sample/iris/test_weight1.h5')

# 6. 모델 사용하기
xhat_idx = np.random.choice(X_test.shape[0], 5)


xhat = X_test[xhat_idx]


yhat = model.predict_classes(xhat)
print(yhat)
for i in range(5):
    print('True : ' + str(np.argmax(Y_test[xhat_idx[i]])) + ', Predict : ' + str(yhat[i]))
