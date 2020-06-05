# import libraries
import numpy as np
import pandas as pds
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes
import pandas as pd
# use diabetes sample data from sklearn
diabetes = load_diabetes()

# load them to X and Y
X = diabetes.data
Y = diabetes.target

print(X.shape)#442,10
print(Y.shape)#442

df = pd.DataFrame(diabetes.data, columns= diabetes.feature_names)
# print(df.tail())
# print(diabetes['DESCR'])
# print(diabetes.keys())
#dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename'])

# create deep learning like regression model
def deep_reg_model():
    model = Sequential()
    model.add(Dense(5, input_dim=10))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(128))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(64))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(32))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(1))


    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# use data split and fit to run the model



# labels = pd.DataFrame(diabetes.target)
# labels.columns=['labels']
# data = pd.DataFrame(diabetes.data)
# data.columns=['age','sex','bmi','bp','s1','s2','s3','s4','s5','s6']
# data = pd.concat([data,labels],axis=1)
# feature = data[['age','sex','bmi','bp','s1','s2','s3','s4','s5','s6']]
# from sklearn.manifold import TSNE
# model = TSNE(learning_rate=50)
# transformed = model.fit_transform(feature)

# xs = transformed[:,0]
# ys = transformed[:,1]





from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
scaler=StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

#
#
from sklearn.decomposition import PCA
pca = PCA(n_components=8)
pca.fit(X)
diabetes_pca = pca.transform(X)
print(diabetes_pca.shape)
print('principal component vec :\n', pca.components_.T)
print('선택한 차원(픽셀) 수 :', pca.n_components_)
print('선택한 차원(픽셀) 수 :', nmf.n


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0, shuffle=False)




# print(diabetes_pca.shape)
# print('principal component vec :\n', pca.components_.T)
# print('선택한 차원(픽셀) 수 :', pca.n_components_)


estimator = KerasRegressor(build_fn=deep_reg_model, epochs=100, batch_size=2, verbose=1)
estimator.fit(x_train, y_train)



y_pred = estimator.predict(x_test)

# show its root mean square error
mse = mean_squared_error(y_test, y_pred)
print("KERAS REG RMSE : %.2f" % (mse ** 0.5))

from sklearn.metrics import r2_score
R2=r2_score(y_test, y_pred)

print(R2)
# '''
# RMSE : 72.26
# '''