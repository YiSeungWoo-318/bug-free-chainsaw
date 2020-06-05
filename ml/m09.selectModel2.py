import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,r2_score
from sklearn.utils.testing import all_estimators
import warnings
from sklearn.datasets import load_boston
warnings.filterwarnings('ignore')

boston = pd.read_csv('./data/csv/boston_house_prices.csv', header=0, index_col=None)

# boston = load_boston()

print(boston)


# boston=np.array(boston)

x=boston.iloc[1:, 0:13]
y=boston.iloc[1:, 13:]

for i in range(len(x)):
    for j in range(len(x.iloc[i])):
         x.iloc[i,j]=float(x.iloc[i,j].replace(',',''))

for i in range(len(y)):
    y.iloc[i,0] = float(y.iloc[i,0].replace(',',''))

x.dropna(axis = 1)
y.dropna(axis = 1)
print(x)
print(x.shape)
print(y)
print(y.shape)




# print(x.shape)
# print(y.shape)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=44)


allAlgorithms = all_estimators(type_filter='regressor')

for (name,algorithm) in allAlgorithms:
    model = algorithm()
    model.fit(x_train, y_train)
    # y_pred = model.predict(x_test)
    acc=model.score(x_test, y_test)
    print(name,"의 정답률=",acc)


# import sklearn
# print(sklearn.__version__)
# acc=model.score(x_test, y_test)
# print(" 의 예측 결과 : ", y_pred)
# # print("acc = ", acc)

# r2=r2_score(y_test, y_pred)
# # print(r2)

# print(name,"의 정답률=",acc)

# print(name,"의 정답률=",r2)