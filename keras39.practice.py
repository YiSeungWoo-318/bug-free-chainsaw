import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM



b = np.array(range(100,110))
a = np.array(range(1,11))
c = np.array(range(1,11))

size =5 

def split_x(seq, size):
    bbb = []
    for i in range(len(seq)-size+1):
         subset = seq[i : (i+size)]
         
         bbb.append([item for item in subset])
    return np.array(bbb)

dataset = split_x(a,size)
print("===============================")
print(dataset)
print(dataset.shape)


def split_xy(seq, size):
    x,y=[],[]
    for i in range(len(seq)):
        T = size+i
        if T > len(seq)-1:
            break
        tmp_x, tmp_y = seq[i:T], seq[T]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy(size,4)
print(x,"\n",y)