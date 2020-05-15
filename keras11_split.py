
import numpy as np

x = np.array(range(1,101))
y = np.array(range(101,201))

x_train = x[:60]
x_val = x[59:80]
x_test = x[79:]

y_train = x[:60]
y_val = x[59:80]
y_test = x[79:]


from keras.models import Sequential
from keras.layers import Dense
model=Sequential()
model.add(Dense(5,input_dim=1)) 
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))


model.compile(loss='mse',optimizer='adam',metrics=['mse']) 
model.fit(x_train,y_train,epochs=500,batch_size=1, validation_data=(x_val, y_val))



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