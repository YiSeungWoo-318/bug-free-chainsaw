#함수 모형으로 구성하라
#1~100, 301~400, 0~99 의 값을 넣어
# 701~800 의 값이 나온다. 임의숫자 x_pred를 넣을 때 y_pred를 구하라!

#0---------------------------------------------------------------------------------------#


import numpy as np
seed=0
np.random.seed(seed)

x=  np.array([range(1,101),range(301,401),range(100)])
y = np.array(range(701,801))

splits=100
x=np.transpose(x)
y=np.transpose(y)

#-----------------------------


from sklearn.model_selection import train_test_split
x_tr,x_te,y_tr,y_te= train_test_split(x, y, test_size =0.25)

#-----------------------------------------------------------------------
#transpose는 행렬의 메소드



from keras.models import Sequential, Model
from keras.layers import Dense, Input

t=Input(shape=(3,)) 
l1 = Dense(50,activation='relu')(t)
l2= Dense(100,activation='relu')(l1)
l3= Dense(50,activation='relu')(l2)
l4= Dense(50,activation='relu')(l3)
l5= Dense(10,activation='relu')(l4)
#함수는 Input과 Output을 해주고 모델도 설정해야 한다
o= Dense(1)(l5)

q =Model(inputs = t, outputs=o)

##-------------------------------------------------------------------

q.compile(loss = 'mse', optimizer= 'adam', metrics=['mse'])
q.fit(x_tr,y_tr,epochs=500, validation_split=0.2, batch_size=2)
X=q.evaluate(x_te,y_te)
print(X)

y_pr=q.predict(x_te)
print(y_pr)



from sklearn.metrics import mean_squared_error as mse, r2_score

def RMSE(a,b):
    return np.sqrt(mse(a,b))

R2=r2_score(y_te,y_pr)

print(RMSE(y_te,y_pr),R2)
