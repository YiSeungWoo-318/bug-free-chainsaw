import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold,cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')

iris = pd.read_csv('./data/csv/iris.csv', header=0)



x= iris.iloc[:,0:4]

y= iris.iloc[:,4]

print(x)
print(y)

kfold = KFold(n_splits=5, shuffle=True)

# warnings.filterwaning('ignore')
# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=44)


# warnings.filterwaning('ignore')
allAlgorithms = all_estimators(type_filter='classifier')

for (name,algorithm) in allAlgorithms:
    model = algorithm()
    scores = cross_val_score(model,x,y,cv=kfold)
    print(name,"의 정답률=")
    print(scores)
    # model.fit(x, y)



import sklearn
print(sklearn.__version__)