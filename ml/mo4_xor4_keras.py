from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsRegressor
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
import numpy as np



#data
x_data = [[0,0],[1,0],[0,1],[1,1]]
y_data= [0,1,1,0]
x_data=np.array(x_data)
y_data= np.array(y_data)
# x_data=x_data.reshape(8,)
from sklearn.model_selection import train_test_split
x_data,x_test,y_data,y_test= train_test_split(x_data,y_data,test_size=0.1)
model=Sequential()
model.add(Dense(5,input_dim=2,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(10,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(10,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(10,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(10,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1,activation='sigmoid'))


model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['acc'])
model.fit(x_data, y_data,batch_size=2, epochs=5000,validation_split=0.1)
e=model.evaluate(x_test,y_test)

# e=model.evaluate(x_data)
# print(e)
y_pred=model.predict(x_data)
print("loss, acc:", e)
print(y_pred)
