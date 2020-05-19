#train을 validation  data로 사용하지 말고 별도의 validation을 만들어서 사용하시오
#데이터에서부터 분류되는 경우


####사실 validation_data를 train으로 쓸 필요 없다
#즉 train으로 훈련한 값을 validation으로 검증 해봤자 똑같은 걸로 똑같이 검증하니 오류도 똑같이 유지된다
# validtion 데이터와 Train 데이터를 분리 하는 것이 맞다datetime A combination of a date and a time. Attributes: ()

#model.fi에서 train을 100번 훈련시킨다고 했을 대 여기서 train을  validation 데이터로 검증하면 비효율적 오히려 acc가 낮아질 것이다. 
import numpy as np

x_train=np.array([1,2,3,4,5,6,7,8,9,10])
y_train=np.array([1,2,3,4,5,6,7,8,9,10])
x_test=np.array([11,12,13,14,15])
y_test=np.array([11,12,13,14,15])  
x_pred=np.array([16,17,18])
x_val=np.array([101, 102, 103, 104, 105])
y_val=np.array([101, 102, 103, 104, 105])



from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(3, input_dim= 1 ))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))


model.compile(loss = 'mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val))

X = model.evaluate(x_test,y_test)
print(X)

y_pred=model.predict(x_pred)
print(y_pred)
'''
from sklearn.metrics import mean_squared_error
def RMSE(x,y):
    return np.sqrt(mean_squared_error(x,y))

print("RMSE : ", RMSE(y_test, y_pred))
'''

###error
#RMSE의 비교군을 y_test라고 했는데 이는 5행으로 되어있고 y_pred 3행으로 되어 있기에 올바른 비교가 되지 않는다. 즉 샘플대치가 안됨
#
from sklearn.metrics import mean_squared_error
def RMSE(x,y):
    return np.sqrt(mean_squared_error(x,y))

print("RMSE : ", RMSE(y_test[0:3], y_pred))
#3개의 값을 뽑음으로 해결!
