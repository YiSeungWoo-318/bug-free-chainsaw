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


# x=np.array(range(1,500))

# def A(seq, size):
#     aaa = []
#     for i in range(len(seq)-size+1):
#         subset = seq[i : (i+size)]
#         aaa.append(i for i in subset )

#     return np.array(aaa)

# dataset = split_x(a,size)    

# def A(a,size):
#     aaa = []
#     for i in range(len(a)-size+1):
#         aa=a[i:(i+size)]
#         aaa.append(i for i in aa)

#     return np.array(aaa)

# D=A(c,size)
# print(D)

