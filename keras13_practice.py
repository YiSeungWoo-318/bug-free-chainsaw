#

##
##
##
# keras를 이용하여 validation을 사용하는 데 validation 이용방법 : 데이터부터 validation을 구분하여 준다. 구분하는 방식에서
## 데이터를 직접 입력해 주는 방법이 있고, 비율을 나누어줘서 줄 수 있다
## 그런데 데이터에서 validation을 구분하지 않는 방법이 있다. 
## 이를 사용하여 13번 문제를 풀어보자.
##
##
#데이터 x의 predict를 구하라.
import numpy as np
x = np.array(range(1, 101))
y = np.array(range(101, 201))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.75)

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))


model.compile( loss = 'mse', optimizer= 'adam', metrics= ['mse'])

model.fit(x_train,y_train,epochs=1000, validation_split=(0.2))
P=model.evaluate(x_test,y_test)
y_p=model.predict(x)

print(y_p)


from sklearn.metrics import mean_squared_error as mse, r2_score
def RMSE(a,b):
    return np.sqrt(mse(a,b))

RMSE_1 = RMSE(y,y_p)
R2= r2_score(y,y_p)
abc = (RMSE_1,R2)
print(" RMSE : ", "R2 : ", RMSE_1, R2 ) 