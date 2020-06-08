import numpy as np
import pandas as pd
import matplotlib as plt

train = pd.read_csv('./data/csv/comp/train.csv', header=0,index_col=0,sep=",")
test = pd.read_csv('./data/csv/comp/test.csv', header=0,index_col=0,sep=",")
sub = pd.read_csv('./data/csv/comp/sample_submission.csv', header=0,index_col=0,sep=",")

# print('train.shape:', train.shape) #10000, 75
# print('test.shape:', test.shape)  #10000, 71
# print('sub.shape:', sub.shape) #10000, 4

# print(train.isnull().sum())

train=train.interpolate()

# print(train.isnull().sum())

test=test.interpolate()


# print(train.head())
# print(train.iloc[0,0])


x=train.iloc[:, :71]
# print(x)
y=train.iloc[:, 71:75]
# print(y)
x=x.values
y=y.values
# print(test.head())

# print(sub.head())


# print(train[:, :])

# print(x.shape)
# print(y.shape)


x=x.reshape(10000,1,71)


from sklearn.model_selection import train_test_split
x_train, x_test = train_test_split(x, random_state=2, train_size=0.8)
from sklearn.model_selection import train_test_split
y_train, y_test = train_test_split(y, random_state=1, train_size=0.8)


#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM
#--------------------------------------#

input1 = Input(shape=(1,71))

dense1_1 = LSTM(5, activation = 'relu',name = 'dense1_1')(input1)
dense1_2 = Dense(5, activation = 'relu',name = 'dense1_2')(dense1_1)
dense1_3 = Dense(5, activation = 'relu',name = 'dense1_3')(dense1_2)
dense1_4 = Dense(5, activation = 'relu',name = 'dense1_4')(dense1_3)
dense1_5 = Dense(3, activation = 'relu',name = 'dense1_5')(dense1_4)

output1 = Dense(2)(dense1_5)
output1_2 = Dense(5)(output1)
output1_3 = Dense(5)(output1_2)
output1_4 = Dense(5)(output1_3)
output1_5 = Dense(4)(output1_4)
#-------------------------------------#


model= Model(inputs=input1, outputs = [output1_5])

model.summary()




model.compile(loss ='mae', optimizer='adam', metrics=['mae'])
model.fit(x_train,y_train, epochs=100, batch_size=1, validation_split=0.2, verbose = 1)


total_loss, loss1, loss2, mse1, mse2 = model.evaluate(x_test, y_test , batch_size=1) 

y_predict = model.predict(test)
print(y_predict)#1


from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict ):
    return np.sqrt(mean_squared_error(y_test, y_predict))  


print("Rmse1 : ", RMSE)#2


from sklearn.metrics import r2_score
r2=r2_score(sub,y_predict)

print(r2_score(sub,y_predict))#3





#서브밋파일을 만든다. 
y_predict.to_csv("./data/y_predict.csv")
