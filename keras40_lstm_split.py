import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input

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

dataset=dataset.reshape(6,5,1)

x=dataset[:,0:4]
y=dataset[:,4]

print(dataset)
'''
#LSTM 모델을 완성하시오

input1= Input(shape=(4,1))
dense1=LSTM(5,name='dense1')(input1)
dense2=Dense(5,name='dense2')(dense1)
dense3=Dense(5,name='dense3')(dense2)
dense4=Dense(5,name='dense4')(dense3)
dense5=Dense(5,name='dense5')(dense4)
dense6=Dense(1)(dense5)
dense7=Dense(1)(dense6)

model = Model(inputs=[input1], outputs=dense7)
model.summary()


model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=5000)
'''