from timeseries import timeseries as ts
import tensorflow as tf
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np

time_series = ts()

# TODO: Pass dataset name as parameter
time_series.readdataset('50words')
time_series.convert_to_GASF()

batch_size = 32
num_classes = time_series.no_classes
epochs = 100

# Using  network architecture similar to alexnet
model = Sequential()
model.add(Conv2D(64, (5, 5), padding='same', input_shape=time_series.gasf_data.shape[1:] + (1,)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
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

model.fit(time_series.gasf_data[:, :, :, np.newaxis], 
          tf.keras.utils.to_categorical((time_series.lables - 1), num_classes=num_classes),
          batch_size=batch_size,
          epochs=epochs,
          shuffle=True)

model.evaluate(x = time_series.gasf_data[:, :, :,np.newaxis], y = time_series.lables)