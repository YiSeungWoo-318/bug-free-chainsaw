from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#data
x_data = [[0,0],[1,0],[0,1],[1,1]]
y_data= [0,1,1,0]

model = SVC()


model.fit(x_data, y_data)

x_test = [0,0],[1,0],[0,1],[1,1]
y_predict = model.predict(x_test)


acc=accuracy_score([0,1,1,0], y_predict)
print(x_test," 의 예측 결과 : ", y_predict)
print("acc = ", acc)