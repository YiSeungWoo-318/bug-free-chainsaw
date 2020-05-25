import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input

#1. 데이터
a = np.array(range(1,101))
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
x_predict=x[90:96, :]


from sklearn.model_selection import train_test_split
x_train,x_test=train_test_split(x,train_size=(0.8))
y_train,y_test=train_test_split(y,train_size=(0.8))
#실습1.train, test 분리
#2. 마지막 6개의 행을 predict로 만들고 싶다
#3. validation을 넣을 것 (train 20%)
print(x.shape)
print(y.shape)
print(x_predict.shape)


# x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],1)
# x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1)

# x_predict=x_predict.reshape(x_predict.shape[0],x_predict.shape[1],1)


# x= np.reshape(x,(4,1))
#x=x.reshape(6,4,1)

model=Sequential()
model.add(Dense(10,activation='relu',input_dim=4))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(5,activation='relu'))
model.add(Dense(1))


model.compile(loss='mse',optimizer='adam',metrics=['mse'])
model.fit(x_train,y_train,epochs=4000,validation_split=(0.2))

loss,mse= model.evaluate(x_test,y_test)
y_predict = model.predict(x_predict)
print('loss : ', loss)
print('mse : ', mse)
print('y_predict : ', y_predict)
