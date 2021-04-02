# study_keras

#### simple train

refer https://blog.csdn.net/mogoweb/article/details/82152174 

```python
## refer https://blog.csdn.net/mogoweb/article/details/82152174
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
np.random.RandomState(0)
trX = np.linspace(-1, 1, 101)
trY = 3*trX + np.random.randn(*trX.shape)*0.33

model = tf.keras.Sequential()
model.add(layers.Dense(input_dim=1, units=1, activation='linear'))

weights = model.layers[0].get_weights()
w_init = weights[0][0][0]
b_init = weights[1][0]
print('Linear regression model is initialized with weights w: %.2f, b: %.2f' % (w_init, b_init))

model.compile(optimizer='sgd', loss='mse') 
model.fit(trX, trY, nb_epoch=600, verbose=1)

weights = model.layers[0].get_weights()
w_final = weights[0][0][0]
b_final = weights[1][0]

print('Linear regression model is trained to have weight w: %.2f, b: %.2f' % (w_final, b_final))

```

多层感知机

```python
## https://cloud.tencent.com/developer/article/1039595
### train a model

from keras.datasets import mnist
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()

ax0=plt.subplot(221)
ax0.imshow(X_train[0])
ax1=plt.subplot(222)
ax1.imshow(X_train[1])
ax2=plt.subplot(223)
ax2.imshow(X_train[2])
ax3 = plt.subplot(224)
ax3.imshow(X_train[3])

import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

seed = 7
np.random.RandomState(seed)

num_pixels = X_train.shape[1]*X_train.shape[2]

print(num_pixels)
print(X_train.shape, X_test.shape)
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

X_train = X_train/255
X_test = X_test/255


y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

def baseline_model():
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='random_uniform', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='random_uniform', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


model = baseline_model()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
scores = model.evaluate(X_test, y_test, verbose=0)

print('Baseline error %s %%'%str(100-scores[1]*100))


```

```shell
keras.layers.Dense(units, 
				  activation=None, 
				  use_bias=True, 
				  kernel_initializer='glorot_uniform', 
				  bias_initializer='zeros', 
				  kernel_regularizer=None, 
				  bias_regularizer=None, 
			      activity_regularizer=None, 
				  kernel_constraint=None, 
				  bias_constraint=None)
units:
该层有几个神经元

activation:
该层使用的激活函数

use_bias:
是否添加偏置项

kernel_initializer:
权重初始化方法

bias_initializer:
偏置值初始化方法

kernel_regularizer:
权重规范化函数

bias_regularizer:
偏置值规范化方法

activity_regularizer:
输出的规范化方法

kernel_constraint:
权重变化限制函数

bias_constraint:
偏置值变化限制函数

https://blog.csdn.net/weixin_42499236/article/details/84624195

model.add(Dense(32, input_shape=(16,)))
# now the model will take as input arrays of shape (*, 16)
# and output arrays of shape (*, 3)

```

CNN 卷积神经网络


```python
# https://cloud.tencent.com/developer/article/1039595
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import tensorflow as tf
#K.set_image_dim_ordering('th')
# =============================================================================
tf.keras.backend.set_image_data_format('channels_last')
# =============================================================================
#设定随机数
seed = 7
numpy.random.seed(seed)
# 加载数据
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# 转换数据 [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0],  28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
#调整输入范围 从0-255 到 0-1
X_train = X_train / 255
X_test = X_test / 255
#独热编码
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


def larger_model():
     # 创建
     model = Sequential()
     #model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
     model.add(Conv2D(filters=30,
                 kernel_size=(5,5),
                 padding='same',
                 input_shape=( 28,28, 1), 
                 activation='relu'))
     model.add(MaxPooling2D(pool_size=(2, 2)))
     
     #model.add(Conv2D(15, (3, 3), activation='relu'))
     model.add(Conv2D(filters=15,
                 kernel_size=(5,5),
                 padding='same',
                 input_shape=(28,28, 1), 
                 activation='relu'))
     model.add(MaxPooling2D(pool_size=(2, 2)))
     model.add(Dropout(0.2))
     model.add(Flatten())
     model.add(Dense(128, activation='relu'))
     model.add(Dense(50, activation='relu'))
     model.add(Dense(num_classes, activation='softmax'))
     # 编译
     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
     return model
 
model = larger_model()
# 调整
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
# 最终输出
scores = model.evaluate(X_test, y_test, verbose=0)
print("Large CNN Error: %.2f%%" % (100-scores[1]*100))    
 

```


keras conv2d 解析

refer https://blog.csdn.net/qq_30527579/article/details/106782019

```python

import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras import layers
from keras.layers.convolutional import  Conv2D

def conv2d_test():
    model = Sequential()
    model.add(Conv2D(filters=3, kernel_size=(2, 2), input_shape=(4,4,1),  kernel_initializer='ones'))
    return model

def test():
    conv2d_model = conv2d_test()
    np.random.RandomState(7)
    #data_test = np.random.randint(1, 3, (1, 4, 4, 1)) one inener 3d , 4x(4,1)  4 ge 2d(4, 1) 
    #numpy.random.randint(low, high=None, size=None, dtype='l') [low, high) 
    #size: int or tuple of ints  ze = (m, n, k)，则输出数组的shape = (m, n, k)
    data_test = np.array([[
                    [[2],[2],[2],[1]],
                    [[2], [1], [2], [2]],
                    [[1], [2], [1], [2]],
                    [[1], [2], [1], [1]]
                    ]])
    predict_result = conv2d_model.predict(data_test)
    print(data_test)
    print(predict_result)
    print(np.shape(predict_result))



if __name__ == '__main__':
    test()
out:
[[[[2]
   [2]
   [2]
   [1]]

  [[2]
   [1]
   [2]
   [2]]

  [[1]
   [2]
   [1]
   [2]]

  [[1]
   [2]
   [1]
   [1]]]]
[[[[7. 7. 7.]
   [7. 7. 7.]
   [7. 7. 7.]]

  [[6. 6. 6.]
   [6. 6. 6.]
   [7. 7. 7.]]

  [[6. 6. 6.]
   [6. 6. 6.]
   [5. 5. 5.]]]]
(1, 3, 3, 3)
```







