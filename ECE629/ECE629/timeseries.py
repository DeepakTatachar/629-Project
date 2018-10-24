import numpy as np
import json
import matplotlib.pyplot as plt
from pyts.image import GASF, MTF
from reader import reader as rdr

class timeseries(object):
   # Stores the entire dataset or size that fits in memory
   dataset = np.empty((1, 1))
   lables = np.empty((1, 1))
   gasf_data = np.empty((1, 1))
   mtf_data = np.empty((1, 1))
   no_classes = 0

   # The GASF and MTF images are of size 128
   image_size = -1

   # Save the location of the dataset in a json so that 
   # I do not have to modify the code every time I want to run it
   # on a different data set, I pass a different argument
   dataset_to_loc_json = './dict.json'

   def __init__(self, image_size = 128):
      self.image_size = image_size

   def readdataset(self, dataset_name):

      # read the json file to get the dictionary
      with open(self.dataset_to_loc_json) as data_file:
         reader_dic = json.load(data_file)

      dataset_metadata = reader_dic[dataset_name]
      reader = rdr(dataset_metadata)
      [self.dataset, self.lables] = reader.read()
      self.no_classes = int(dataset_metadata['no_classes'])

   def convert_to_GASF(self):
      gasf = GASF(self.image_size)
      self.gasf_data = gasf.fit_transform(self.dataset)

   def convert_to_MTF(self):
      mtf = MTF(self.image_size)
      self.mtf_data = mtf.fit_transform(self.dataset)



