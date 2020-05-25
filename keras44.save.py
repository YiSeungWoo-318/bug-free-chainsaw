import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input


#2.모델
model=Sequential()
model.add(LSTM(10,activation='relu',input_shape=(4,1)))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(5,activation='relu'))
model.add(Dense(10))


model.summary()

# model.save(".//keras//save_keras44.h5")
model.save(".\model\save_keras44.h5")
print('저장잘됬다.')