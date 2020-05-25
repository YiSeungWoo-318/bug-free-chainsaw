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

x=dataset[:,0:4]
y=dataset[:,4]

x= np.reshape(x,(6,4,1))
#x=x.reshape(6,4,1)


model=Sequential()
model.add(LSTM(5,input_shape=(4,1)))
model.add(Dense(3))
model.add(Dense(1))
model.summary()




from keras.callbacks import EarlyStopping, TensorBoard
tb_hist = TensorBoard(log_dir='graph',histogram_freq=0, write_graph=True, write_images=True)




model.compile(loss='mse',optimizer='adam',metrics=['acc'])
hist=model.fit(x,y,epochs=4000,validation_split=(0.2),callbacks=[tb_hist])
print(hist)
print(hist.history.keys())
import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('loss,acc')
plt.ylabel('loss,acc')
plt.xlabel('epoch')
plt.legend(['train loss', 'test loss', 'train acc', 'test acc'])
# plt.show()

# loss,mse= model.evaluate(x,y)
# y_predict = model.predict(x)
# print('loss : ', loss)
# print('mse : ', mse)
# print('y_predict : ', y_predict)