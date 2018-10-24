from timeseries import timeseries as ts
import tensorflow as tf

time_series = ts()

# TODO: Pass dataset name as parameter
time_series.readdataset('50words')

gasf_data = time_series.getGASF()

model = tf.keras.models.Sequential()

