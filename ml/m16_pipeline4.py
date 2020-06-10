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




parameters ={'kerasclassifier__hidden_layers' : [1,2,3,5,7,9], 

             'kerasclassifier__nodes' : [64,128,256,512], 

             'kerasclassifier__activation' : ['relu','linear'], 

             'kerasclassifier__drop' : [0.1,0.2,0.3,0.4,0.5], 

             'kerasclassifier__optimizer' : ['adam'],

             "kerasclassifier__batch_size" : [50,10],

             'kerasclassifier__epochs' : [10,50]}


from keras.wrappers.scikit_learn import KerasClassifier
m = KerasClassifier(build_fn=build_model, epochs=1000, verbose=1)
# hyperparameters = create_hyperparameters()


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# search = RandomizedSearchCV(model, hyperparameters, cv=3, n_jobs=1)
# search.fit(x_train, y_train)

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# pipe = Pipeline([("scaler",MinMaxScaler()),('svm', SVC())])
pipe = make_pipeline(MinMaxScaler(), m)
model = RandomizedSearchCV(pipe, parameters, cv=5)




pipe.fit(x_train, y_train)
score=pipe.score(x_test,y_test)
y_pred=pipe.predict(x_test)
print(y_pred)
print(score)
print("parameter:", model.best_params_)
print("estimator:",model.best_estimator_)
from sklearn.metrics import mean_squared_error, r2_score
def RMSE():
    return np.sqrt(mean_squared_error(y_test,y_pred))
# print("loss : ", RMSE(y_test, y_pred))
# print("r2:", r2_score(y_test,y_pred))