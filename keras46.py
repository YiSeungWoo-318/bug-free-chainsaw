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



from keras.models import load_model
model = load_model('./model/save_keras44.h5')

from keras.layers import Dense
model.add(Dense(1,name='dense_x'))
model.summary()



model.compile(loss='mse',optimizer='adam',metrics=['acc'])
hist=model.fit(x,y,epochs=4000,validation_split=(0.2))
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
plt.show()

loss,mse= model.evaluate(x,y)
y_predict = model.predict(x)
print('loss : ', loss)
print('mse : ', mse)
print('y_predict : ', y_predict)
