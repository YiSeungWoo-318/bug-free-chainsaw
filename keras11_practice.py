#x의 값으로 y를 예측하시오
#train과 test 와 validation을 사용하고 비율로 나눌 것

#데이터
import numpy as np

x = np.array(range(1,101))
y = np.array(range(101,201))


x_train= x[40:]
y_train = y[40:]

x_test = x[20:40]
y_test = y[20:40]

x_val = x[0:20]
y_val = x[0:20]
print(x_train, x_test, x_val)


#model

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(5))
model.add(Dense(1))


model.compile(loss= 'mse', optimizer='adam', metrics=['mse'])
model.fit(x_train,y_train, epochs=1000, validation_data = (x_val, y_val))
X = model.evaluate(x_test, y_test)

y_pred=model.predict(x)
print(y_pred)

from sklearn.metrics import mean_squared_error as mse

def RMSE(a,b):
    return np.sqrt(mse(a,b))


print("RMSE는 : ", RMSE(y,y_pred))

from sklearn.metrics import r2_score

r2=r2_score(y,y_pred)

print("R2:",r2)
