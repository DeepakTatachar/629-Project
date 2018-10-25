import numpy as np
from enum import Enum
import numpy as np

class DataType(Enum):
   txt = 1

class reader(object):
   data_type = DataType.txt
   path = './'
   train_filename = ''
   test_filename = ''

   def __init__(self, metadata):
      self.data_type = DataType[metadata['type']]
      self.path = metadata['path']
      self.train_filename = metadata['train_filename']
      self.test_filename = metadata['test_filename']

   def read(self, train=True):
      if(self.data_type == DataType.txt):
         return self.__read_txt(train)
      return

   def __read_txt(self, train=True):
      if(train):
         file = open(self.path + self.train_filename,"r")
      else:
         file = open(self.path + self.test_filename,"r")

      data_as_string = file.readlines()
      dataset = []
      for line in data_as_string:
         line_data = [float(i) for i in line.split()];
         dataset.append(line_data)

      dataset = np.array(dataset)

      # This is specific to 50words
      data = dataset[:, 1:]
      labels = dataset[:, 0]
      return (data, labels)




      

