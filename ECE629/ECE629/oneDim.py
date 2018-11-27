from timeseries import timeseries as ts
import tensorflow as tf
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from datetime import datetime
import numpy as np

time_series = ts(image_size=96)

time_series.readdataset('Adiac')


train_x = time_series.train_data
test_x = time_series.test_data

batch_size = 32
num_classes = time_series.no_classes
epochs = 5

# Using  network architecture similar to alexnet

# Parameter equivalent dense layer Conv2D(32, (8, 8), padding='same', input_shape=test_x.shape[1:])
# Channels * filter * receptive field size
l1_eqv = 2 * 32 * 8 * 8
model = Sequential()
model.add(Dense(l1_eqv, input_shape=test_x.shape[1:]))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Parameter equivalent dense layer Conv2D(64, (3, 3), padding='same')
# Channels * filter * receptive field size
l2_eqv = 64 * 3 * 3
model.add(Dense(l2_eqv))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Parameter equivalent dense layer Conv2D(128, (3, 3), padding='same')
# Channels * filter * receptive field size
l3_eqv = 128 * 3 * 3
model.add(Dense(l3_eqv))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

startTime = datetime.now()

model.fit(train_x, 
          tf.keras.utils.to_categorical((time_series.train_lables-1), num_classes=num_classes),
          batch_size=batch_size,
          epochs=epochs,
          shuffle=True)

score = model.evaluate(x = test_x, y = tf.keras.utils.to_categorical((time_series.test_lables-1)))

print(datetime.now() - startTime)
print(score)
