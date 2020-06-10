import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input,Conv1D

#1. 데이터
a = np.array(range(1,11))
size = 5 #time_steps=4

def split_x(seq, size):
    bbb = []
    for i in range(len(seq)-size+1):
         subset = seq[i : (i+size)]
         
         bbb.append([item for item in subset])
    print(type(bbb))
    return np.array(bbb)

dataset = split_x(a,size)

print(dataset)
print(dataset.shape)
print(type(dataset))

x=dataset[:,0:4]
y=dataset[:,4]

# x= np.reshape(x,(6,4,1))
x=x.reshape(6,4,1)

model=Sequential()
model.add(Conv1D(140,2,input_shape=(4,1)))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

model.summary()


# from keras.callbacks import EarlyStopping
# earlystopping = EarlyStopping(monitor='loss', patience=100, mode ='min')
# model.compile(loss='mse',optimizer='adam',metrics=['mse'])
# model.fit(x,y,epochs=1000,callbacks =[earlystopping])

# loss,mse= model.evaluate(x,y)
# y_predict = model.predict(x)
# print('loss : ', loss)
# print('mse : ', mse)
# print('y_predict : ', y_predict)