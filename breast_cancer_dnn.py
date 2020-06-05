from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve

whole_data = load_breast_cancer()

X_data = whole_data.data
y_data = whole_data.target
print(X_data.shape)#569,30


from keras.utils import np_utils

y_data = np_utils.to_categorical(y_data)
print(y_data.shape)#569
df = pd.DataFrame(whole_data.data, columns= whole_data.feature_names)
# print(df.tail())
# print(whole_data['DESCR'])
# print(whole_data.keys())

validation_size=0.2
seed=12
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = validation_size, shuffle=True)

num_fold=10
kfold=KFold(n_splits=10, random_state=12)
seed=12

nms = MinMaxScaler()
X_trainn=nms.fit_transform(X_train)
X_testn=nms.fit_transform(X_test)
from keras.models import Sequential
model = Sequential()
from keras.layers import Activation, Dense
model.add(Dense(50, activation='sigmoid', input_shape = (30,)))

model.add(Dense(100, activation='sigmoid'))

model.add(Dense(150, activation='sigmoid'))

model.add(Dense(100, activation='sigmoid'))

model.add(Dense(50, activation='sigmoid'))

model.add(Dense(2, activation='sigmoid'))

import numpy as np


model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])
model.fit(X_trainn, y_train, batch_size = 2, epochs = 90, verbose = 1, validation_split=0.2)
yhat=model.evaluate(X_testn,y_test)


print([np.argmax(yhat, axis=None, out=None)])

# xhat_idx = np.random.choice(X_testn.shape[0], 5)


# xhat = X_testn[xhat_idx]


# yhat = model.predict_classes(xhat)
# print(yhat)

# for i in range(5):
#     print('True : ' + str(np.argmax(y_test[xhat_idx[i]])) + ', Predict : ' + str(yhat[i]))