import numpy as np
# dataset = np.array([1,2,3,4,5,6,7,8,9,10])

# def split_xy1(dataset, time_steps):
#     x, y = list(), list()
#     for i in range(len(dataset)):
#         end_number = i + time_steps
#         if end_number > len(dataset) -1:
#             break
#         tmp_x, tmp_y = dataset[i:end_number],dataset[end_number]
#         print(tmp_x)
#         print(tmp_y)
#         x.append(tmp_x)
#         y.append(tmp_y)
#     return np.array(x), np.array(y)

# x, y = split_xy1(dataset,4 )
# print(x,"\n",y)
# print(x.shape)
# print(y.shape)

###########행렬을 마음대로 자르는 것을 연습해보장~~

M=np.array([2,52,456,7,2,56,7,35,3476,74,543,76,856,654,634,5,4,678,89])
N=np.array([[2,52],[456,7],[2,56],[7,35],[3476,74],[543,76],[856,654],[634,5],[4,678],[89,87]])
#2열로 만들어보장
# print(N.shape)
# print(M)
size=2
def split_x(seq, size):
    bbb = []
    for i in range(len(seq)):#전체
        if i%2==0:
             continue
        subset = seq[i : (i+size)]#19,19+size
        bbb.append([item for item in subset])         
    return list(bbb)

dataset = split_x(M,size)
# print("===============================")
print(dataset)
Dataset=np.array(dataset)
print(Dataset)
print(Dataset.shape)
# dataset = dataset.reshape(19,1)
# print(dataset)




# def split_cc(batch_size , time_steps):
#     x, y = []
#     for i in range(len(batch_size)):
#         C=i+time_steps 
#         if C>2:
#             break
#         tmp_x, tmp_y = batch_size[i:C],batch_size[C]
#         x.append(tmp_x)   
#         y.append(tmp_y)
#     return np.array(x), np.array(y)

# def split_x(seq, size):
#     bbb = []
#     for i in range(len(seq)-size+1):
#          subset = seq[i : (i+size)]
         
#          bbb.append([item for item in subset])
#     return np.array(bbb)

# dataset = split_x(a,size)
# print("===============================")
# print(dataset)def split_x(seq, size):
#     bbb = []
#     for i in range(len(seq)-size+1):
#          subset = seq[i : (i+size)]
         
#          bbb.append([item for item in subset])
#     return np.array(bbb)

# dataset = split_x(a,size)
# print("===============================")
# print(dataset)

# x, y = split_cc(batch_size,4)
# print(x,"\n",y)

