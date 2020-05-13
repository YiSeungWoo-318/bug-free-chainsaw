#1. 데이터

import numpy as np

x_train=np.array([1,2,3,4,5,6,7,8,9,10])
y_train=np.array([1,2,3,4,5,6,7,8,9,10])
x_test=np.array([11,12,13,14,15])
y_test=np.array([11,12,13,14,15])  
x_pred=np.array([16,17,18])

#2. 모델구성

from keras.models import Sequential
from keras.layers import Dense
model=Sequential()
model.add(Dense(5,input_dim=1)) 
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))



#3.훈련
model.compile(loss='mse',optimizer='adam',metrics=['mse']) 
model.fit(x_train,y_train,epochs=1000,batch_size=1)



#4.평가

loss,mse=model.evaluate(x_test,y_test)
print("loss:",loss)
print("mse:",mse)

y_predict=model.predict(x_test)
print(y_predict)

from sklearn.metrics import mean_squared_error as mse

def RMSE(y_test,y_predict):
    return np.sqrt(mse(y_test,y_predict))

print("RMSE:",RMSE(y_test,y_predict))

from sklearn.metrics import r2_score

r2=r2_score(y_test,y_predict)

print("R2:",r2)

#즉, RMSE는 낮게 R2는 높게