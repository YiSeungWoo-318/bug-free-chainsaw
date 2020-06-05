import numpy as np


#데이터
x = np.array(range(1,11))
y= np.array([1,0,1,0,1,0,1,0,1,0])


#모델


from keras.models import Sequential
from keras.layers import Dense


model = Sequential()
model.add(Dense(5,activation="sigmoid", input_dim =1))
model.add(Dense(100,activation="sigmoid"))
model.add(Dense(100,activation="sigmoid"))
model.add(Dense(100,activation="sigmoid"))
model.add(Dense(50,activation="sigmoid"))
model.add(Dense(5,activation="sigmoid"))
model.add(Dense(1,activation="sigmoid"))



#
model.compile(loss='binary_crossentropy', optimizer= 'adam', metrics = ['acc'])
model.fit(x,y,epochs=3000)

y1=model.predict(x)
print(y1)

from sklearn.metrics import mean_squared_error as mse

def RMSE(y_test,y_predict):
    return np.sqrt(mse(y_test,y_predict))

print("RMSE:",RMSE(y,y1))

from sklearn.metrics import r2_score

r2=r2_score(y,y1)

print("R2:",r2)