import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils.np_utils import to_categorical

dataset_original = pd.read_csv('./XYZ.csv')

# Dron NaN value from Data Frame
dataset = dataset_original.dropna()


# data cleansing
X = dataset.iloc[:, 0:78]
print(X.info())
print(type(X))
y = dataset.iloc[:, 78] #78 is labeled column contains 4 anomaly type
print(y)


# encode the labels to 0, 1 respectively
print(y[100:110])
encoder = LabelEncoder()
y = encoder.fit_transform(y)
print([y[100:110]])

# Split the dataset now
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=0)


# feature scaling
scalar = StandardScaler()
XTrain = scalar.fit_transform(XTrain)
XTest = scalar.transform(XTest)


# modeling
model = Sequential()
model.add(Dense(units=16, kernel_initializer='uniform', activation='relu', input_dim=78))
model.add(Dense(units=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(XTrain, yTrain, batch_size=1000, epochs=10)


history = model.fit(XTrain, yTrain, batch_size=1000, epochs=10, verbose=1, validation_data=(XTest, 
yTest))



yPred = model.predict(XTest)
yPred = [1 if y > 0.5 else 0 for y in yPred]
matrix = confusion_matrix(yTest, yPred)#`enter code here`
print(matrix)
accuracy = (matrix[0][0] + matrix[1][1]) / (matrix[0][0] + matrix[0][1] + matrix[1][0] + matrix[1][1])
print("Accuracy: " + str(accuracy * 100) + "%")