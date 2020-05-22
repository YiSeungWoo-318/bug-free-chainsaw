import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM



b = np.array(range(100,110))
a = np.array(range(1,11))


size =5 

def split_x(seq, size):
    bbb = []
    for i in range(len(seq)-size+1):
         subset = seq[i : (i+size)]
         
         bbb.append([item for item in subset])
    print(bbb)
    return np.array(bbb)

dataset = split_x(a,size)
print("===============================")
print(dataset)