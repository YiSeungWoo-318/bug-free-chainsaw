# import pandas as pd
# import numpy as np
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
# from sklearn.svm import SVC




# iris=load_iris()
# x=iris.data
# y=iris.target


# x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.2, shuffle=True, random_state=43)


# #그리드/랜덤 서치에서 사용할 매개변수

# parameters = [
#     {"svm_C" : [1,10,100,1000], "svm_kernel":['linear']},
#     {"svm_C" : [1,10,100,1000], "svm_kernel":['rbf'],
#                                 'svm_gamma':[0.001,0.0001]},
#     {"svm_C" : [1,10,100,1000], "svm_kernel":['sigmoid'], 
#                                 'svm_gamma':[0.001,0.0001]}
    
    

# ]



# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import MinMaxScaler, StandardScaler

# pipe = Pipeline([("scaler",MinMaxScaler()),('svm', SVC())])

# model=RandomizedSearchCV(pipe,parameters,cv=5)

# model.fit(x_train,y_train)

# acc=model.score(x_test,y_test)



# print("acc : ", model.best_estimator_)

# print("acc :", acc )






















# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.model_selection import RandomizedSearchCV
# from keras.wrappers.scikit_learn import KerasClassifier

# #1. 데이터 

# dataset= load_iris()
# x = dataset.data
# y = dataset.target

# x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,shuffle= True)

# # 그리드/랜덤 매개변수
# param = [
#     {"C" : [1,10,100,1000],"kernel": ["linear","rbf","sigmoid"]},
#     {"C" : [1,10,100,1000],"kernel": ["rbf"],"gamma" : [0.001,0.0001]},
#     {"C" : [1,10,100,1000],"kernel": ["sigmoid" ],"gamma" : [0.001,0.0001]}
# ]


# #2 모델

# # model = SVC()

# pipe = Pipeline([("scaler", MinMaxScaler()),("svc",SVC())])

# model = RandomizedSearchCV(pipe,param_distributions=param,cv=5)
# model.fit(x_train, y_train)

# print("best para  :  ",model.best_params_)
# print("acc : ",model.score(x_test,y_test))
# print("acc : ",model.best_score_)

# RandomizedSearchCV + Pipeline

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# 1. 데이터
iris = load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle=True, random_state=43)

# 그리드 / 랜덤 서치에서 사용할 매개 변수
parameters = [
    {"svm__C" :[1, 10, 100, 1000], "svm__kernel" :['linear']},
    {"svm__C" :[1, 10, 100,1000], "svm__kernel" :['rbf'], 'svm__gamma':[0.001, 0.0001]},
    {"svm__C" :[1, 10, 100, 1000], "svm__kernel" :['sigmoid'], 'svm__gamma':[0.001, 0.0001]}
]

# 2. 모델
# model = SVC()

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# pipe = make_pipeline(MinMaxScaler(),SVC())
pipe = Pipeline([("scaler", MinMaxScaler()), ('svm', SVC())])

model = RandomizedSearchCV(pipe, parameters, cv=5)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
acc = model.score(x_test, y_test)

print("최적의 매개변수 = ", model.best_estimator_)
print("acc : ", acc)



# pipe.fit(x_train, y_train)

# print("acc : ", pipe.score(x_test, y_test))

import sklearn as sk
print("sklearn:", sk.__version__)
