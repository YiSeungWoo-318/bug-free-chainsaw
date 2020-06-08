# from keras.datasets import mnist
# from keras.utils import np_utils
# from keras.models import Sequential, Model

# from keras.layers import Input, Dropout, Conv2D, Flatten
# from keras.layers import Dense, MaxPooling2D
# import numpy as np

# (x_train, y_train), (x_test, y_test)= mnist.load_data()

# print(x_train.shape)
# print(x_test.shape)

# x_train = x_train.reshape(x_train.shape[0],28*28)/255
# x_test = x_test.reshape(x_test.shape[0],28*28)/255


# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)

# print(y_train.shape)

# def build_model(drop=0.5, optimizer='adam'):
#     inputs = Input(shape=(28*28), name='input')
#     x=Dense(512,activation='relu', name='hidden1')(inputs)
#     x=Dropout(drop)(x)
#     x=Dense(256, activation='relu',name='hidden2')(x)
#     x=Dropout(drop)(x)
#     x=Dense(128, activation='relu',name='hidden3')(x)
#     x=Dropout(drop)(x)
#     outputs = Dense(10, activation='softmax', name='outputs')(x)
#     model = Model(inputs = inputs , outputs = outputs)
#     model.compile(optimizer=optimizer, metrics=['acc'], loss = 'categorical_crossentropy')
#     return model 

# def create_hyperparameters():
#     batches = [10,20,30,40,50]
#     optimizers = ['rmsprop', 'adam', 'adadelta']
#     dropout = np.linespace(0.1,5,0.5)
#     return{"batch_size": batches, "optimizer": optimizers, "drop" : dropout} 


    
# from keras.wrappers.scikit_learn import KerasClassifier
# model=kerasClassifier(build_fn= build_model, verbose=1)
# hyperparameters=create_hyperparameters()

# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# search = GridSearchCV(estimator=model, hyperparameters, cv=3)
# search.fit(x_train, y_train)
# print(search.best_params_)


from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Conv2D, Flatten, Dense
from keras.layers import MaxPooling2D
import numpy as np


# 1. 데이터
(x_train, y_train),(x_test, y_test) = mnist.load_data()

print(x_train.shape)    # (60000, 28, 28)
print(x_test.shape)     # (10000, 28, 28)

# x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)/255
# x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)/255

x_train = x_train.reshape(x_train.shape[0], 28*28)/255
x_test = x_test.reshape(x_test.shape[0], 28*28)/255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(y_train.shape)

# 2. 모델

def build_model(drop=0.5, optimizer='adam'):
    inputs = Input(shape=(28*28, ), name='input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs = inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'], loss='categorical_crossentropy')

    return model

def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5)
    return{"batch_size" : batches, "optimizer" : optimizers,
            "drop" : dropout}


from keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn=build_model, verbose=1)

hyperparameters = create_hyperparameters()

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = GridSearchCV(model, hyperparameters, cv=3)
search.fit(x_train, y_train)

print(search.best_params_)
