import numpy as np
import pandas as pd

datasets = pd.read_csv("./data/csv/iris.csv", index_col=None,header=0,sep=',')

print(datasets)
print(datasets.head())
print(datasets.tail())


print("========================")

#numpy로 바꾼다.





iris_load = np.save('./data/iris.npy',arr=datasets)