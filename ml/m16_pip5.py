from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Conv2D, Flatten, Dense
from keras.layers import MaxPooling2D
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 1. 데이터
iris=load_iris()
x=iris.data
y=iris.target


x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.2, shuffle=True, random_state=43)

# 2. 모델

def build_model(drop=0.5, optimizer='adam'):
    inputs = Input(shape=(4, ), name='input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(3, activation='softmax', name='outputs')(x)
    model = Model(inputs = inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'], loss='categorical_crossentropy')

    return model

def create_hyperparameters():
    
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.2, 0.3]
    
    return{"kerasclassifier__batch_size" : batches, "kerasclassifier__optimizer" : optimizers,
            "kerasclassifier__drop" : dropout}


from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn=build_model, verbose=1)
hyperparameters = create_hyperparameters()

# pipe = Pipeline([("scaler",MinMaxScaler()),('models', model)])
pipe = make_pipeline(MinMaxScaler(), model)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(pipe, hyperparameters, cv=3, n_jobs=1)
search.fit(x_train, y_train)
search.score(x_test,y_test)
print('최적의 매개변수:',search.best_estimator_)
print('최적의 파라미터:',search.best_params_)
print("acc : ", search.score(x_test,y_test))