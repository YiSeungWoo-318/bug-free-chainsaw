import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split, cross_val_score
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
from keras.layers import Dense, Input, LSTM, BatchNormalization,Dropout
#--------------------------------------#
def build_model():
    input1 = Input(shape=(1, 71))

    dense1_1 = LSTM(5, activation='relu', name='dense1_1')(input1)
    d = Dropout(0.1)(dense1_1)
    b = BatchNormalization()(d)
    dense1_2 = Dense(5, activation='relu', name='dense1_2')(b)
    d1 = Dropout(0.1)(dense1_2)
    b1 = BatchNormalization()(d1)
    dense1_3 = Dense(5, activation='relu', name='dense1_3')(b1)
    d2 = Dropout(0.1)(dense1_3)
    b2 = BatchNormalization()(d2)
    dense1_4 = Dense(5, activation='relu', name='dense1_4')(b2)
    d3 = Dropout(0.1)(dense1_4)
    b3 = BatchNormalization()(d3)
    dense1_5 = Dense(3, activation='relu', name='dense1_5')(b3)
    d4 = Dropout(0.1)(dense1_5)
    b4 = BatchNormalization()(d4)

    output1 = Dense(2)(b4)
    d5 = Dropout(0.1)(output1)
    b5 = BatchNormalization()(d5)
    output1_2 = Dense(5)(b5)
    d6 = Dropout(0.1)(output1_2)
    b6 = BatchNormalization()(d6)
    output1_3 = Dense(5)(b6)
    d7 = Dropout(0.1)(output1_3)
    b7 = BatchNormalization()(d7)
    output1_4 = Dense(5)(b7)
    d8 = Dropout(0.1)(output1_4)
    b8 = BatchNormalization()(d8)
    output1_5 = Dense(4)(b8)

    # -------------------------------------#

    model = Model(inputs=input1, outputs=[output1_5])
    model.compile(loss='mae', optimizer='adam', metrics=['mae'])
    return model
def create_hyperparameters():
    batches = [2,10,32,40]
    epochs=[100,200,300,400,500,600,700,1000]
    return{"batch_size" : batches, "epochs" : epochs}


from keras.wrappers.scikit_learn import KerasRegressor
model = KerasRegressor(build_fn=build_model(), verbose=1)

hyperparameters = create_hyperparameters()

from sklearn.model_selection import RandomizedSearchCV
search = RandomizedSearchCV(model, hyperparameters, cv=10)
search.fit(x_train, y_train)

print(search.best_params_)