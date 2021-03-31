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
