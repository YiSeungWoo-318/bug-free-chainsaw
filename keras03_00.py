#1.데이터
import numpy as np 
x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([1,2,3,4,5,6,7,8,9,10])


#2.모델구성
from keras.models import Sequential
from keras.layers import Dense
model_youngsun = Sequential()

model_youngsun.add(Dense(5, input_dim=1))
model_youngsun.add(Dense(5))
model_youngsun.add(Dense(5))
model_youngsun.add(Dense(1)) 


#3.훈련
model_youngsun.compile(loss='mse', optimizer='adam', metrics=['acc'])
model_youngsun.fit(x,y,epochs=100, batch_size=1)
#4.평가, 예측 
loss, acc = model_youngsun.evaluate(x,y,batch_size=1)
print("loss:",loss)
print("acc:",acc)
