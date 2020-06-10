import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier


iris=load_iris()
x=iris.data
y=iris.target


x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.2, shuffle=True, random_state=43)



parameters = [
    {"randomforestclassfier__n_jobs" :[1,2,-1]},
    {"randomforestclassfier__n_estimator" :[range(1,100,1)]},
    # {"svc__C" :[1, 10, 100,1000], "svc__kernel" :['rbf'], 'svc__gamma':[0.001, 0.0001]},
    # {"svc__C" :[1, 10, 100, 1000], "svc__kernel" :['sigmoid'], 'svc__gamma':[0.001, 0.0001]}
]


from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# pipe = Pipeline([("scaler",MinMaxScaler()),('svm', SVC())])
pipe = make_pipeline(MinMaxScaler(), RandomForestClassifier())
model = RandomizedSearchCV(pipe, parameters, cv=5)

pipe.fit(x_train, y_train)


print("acc : ", pipe.score(x_test,y_test))

import sklearn as sk
# print("sk:", __sk