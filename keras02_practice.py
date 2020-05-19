
# 두 데이터 값을 훈련시킨 뒤 하나의 값에서 예측하기, validation 사용


import numpy as np

x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([101,102,103,104,105,106,107,108,109,110])
y_test = np.array([101,102,103,104,105,106,107,108,109,110])

# 순차모델로 해결
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense( 5, input_dim = 1))
model.add(Dense  (5))
model.add(Dense (5))
model.add(Dense (1))

model.compile(loss="mse", optimizer="adam", metrics= ["acc"])
model.fit(x_train, y_train , epochs = 100, validation_data= (x_train, y_train))
X=model.evaluate(x_test, y_test)

y_predict=model.predict(x_test)

print(y_predict)