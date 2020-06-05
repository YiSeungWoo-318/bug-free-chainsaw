from sklearn.datasets import load_boston
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D
from keras.layers import Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt
import numpy as np
# (x_train, y_train), (x_test,y_test)=load_boston.load_data()
'''
data : x값
target : y값
'''
dataset=load_boston()
x=dataset.data
y=dataset.target
'''
x과 (?,13)
'''
# P=dataset.data.max
# print(P)
# print(dataset.data.max)
# print(dataset.target.min)
# max구하는 방법?
# minmax를 쓰지 않고 최대값으로 나누어 일로 만들어준다. 
#  x-최소
# 최대-최소
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
scaler=StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
#fit( )과 transform( ) 을 호출하여 PCA 변환 데이터 반환
pca.fit(x)
iris_pca = pca.transform(x)
print(iris_pca.shape)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75)


x_train = x_train.reshape(379,13)
x_test = x_test.reshape(127,13)

model = Sequential()
model.add(Dense(100,input_shape=(13,),activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='elu'))
model.add(Dense(50,activation='elu'))
model.add(Dense(1))
model.summary()

from keras.callbacks import ModelCheckpoint
modelpath = './model/sample/boston/check-{epoch:02d}-{val_loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath=modelpath,monitor='val_loss', save_best_only=True, save_weights_only=False, verbose=1)

model.compile(loss='mse', optimizer='adam', metrics = ['mse'])
hist=model.fit(x_train, y_train, epochs=10000, callbacks=[checkpoint],validation_split=(0.3))
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=14)

model.save('./model/sample/boston/model_test01.h5')
model.save_weights('./model/sample/boston/test_weight1.h5')


y_predict=model.predict(x_test)
#------------------------------------------------------------------------------------------------------------------#
'''
1.pca구하는 법???
2.데이터max구하는법?
3.분류와 회귀 구별
4.
'''
from sklearn.metrics import mean_squared_error as mse, r2_score

def RMSE(y_test,y_predict):
    return np.sqrt(mse(y_test,y_predict))

print("RMSE:",RMSE(y_test,y_predict))

r2=r2_score(y_test,y_predict)
print(r2)
