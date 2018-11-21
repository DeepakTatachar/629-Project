import numpy as np
import json
import matplotlib.pyplot as plt
from pyts.image import GASF, MTF
from reader import reader as rdr
import math
from skimage.transform import rescale

class timeseries(object):

   # Holds the training data for the timeseries (2D array, each row is a sample)
   train_data = np.empty((1, 1))

   # Holds the training labels for the timeseries
   train_lables = np.empty((1, 1))

   # Holds the training data converted to an image
   train_mtf_data = np.empty((1, 1))
   train_gasf_data = np.empty((1, 1))

   # Similar to training data but holds testing data
   test_data = np.empty((1, 1))
   test_lables = np.empty((1, 1))
   test_gasf_data = np.empty((1, 1))
   test_mtf_data = np.empty((1, 1))

   no_classes = 0

   # The GASF and MTF images are of size 128
   image_size = -1
   GASF_image_size = 128

   # Save the location of the dataset in a json so that 
   # I do not have to modify the code every time I want to run it
   # on a different data set, I will pass a different argument
   dataset_to_loc_json = './dict.json'

   def __init__(self, image_size = 128):
      self.GASF_image_size = image_size

   def readdataset(self, dataset_name):

      # Read the json file to get the dictionary
      with open(self.dataset_to_loc_json) as data_file:
         parsedJSON = json.load(data_file)

      reader = rdr(parsedJSON, dataset_name)

      # Read the training data and the testing data
      [self.train_data, self.train_lables] = reader.read()
      [self.test_data, self.test_lables] = reader.read(train=False)

      self.no_classes = int(parsedJSON[dataset_name]['no_classes'])
      self.image_size = math.floor(math.sqrt(self.train_data.shape[1:][0]))

   def convert_to_GASF(self):
      gasf = GASF(self.GASF_image_size)
      self.train_gasf_data = gasf.fit_transform(self.train_data)
      self.test_gasf_data = gasf.fit_transform(self.test_data)

   def convert_to_MTF(self):
      mtf = MTF(self.image_size)
      train_mtf_data = mtf.fit_transform(self.train_data)
      test_mtf_data = mtf.fit_transform(self.test_data)
      self.train_mtf_data = np.empty((train_mtf_data.shape[0], self.GASF_image_size, self.GASF_image_size))
      self.test_mtf_data = np.empty((test_mtf_data.shape[0], self.GASF_image_size, self.GASF_image_size))

      for i in range(train_mtf_data.shape[0]):
         self.train_mtf_data[i:, : ,:] = rescale(train_mtf_data[i, :, :], self.GASF_image_size / self.image_size)

      for i in range(test_mtf_data.shape[0]):
         self.test_mtf_data[i:, : ,:] = rescale(test_mtf_data[i, :, :], self.GASF_image_size / self.image_size)



