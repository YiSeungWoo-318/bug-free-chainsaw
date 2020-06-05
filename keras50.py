from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D

model = Sequential()
model.add(Conv2D(10,(2,2),input_shape=(5,5,1)))#10=fileter, ((2,2)=kernel size,  kernel size=2) height,width,channel 행가로세로 색깔
model.add(Conv2D(7,(2,2)))
model.add(Conv2D(5,(2,2),padding='same'))
model.add(Conv2D(5,(2,2)))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(1))
model.summary()
#스트라이드 default =1 1칸씩움직임 
#valid가 기본
#maxpooling 즁요한 특성을 뽑아냄
model.add(MaxPooling2D())