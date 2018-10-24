import numpy as np
from enum import Enum
import numpy as np

class DataType(Enum):
   txt = 1

class reader(object):
   data_type = DataType.txt
   path = './'
   filename = ''

   def __init__(self, metadata, train=True):
      self.data_type = DataType[metadata['type']]
      self.path = metadata['path']
      if(train):
         self.filename = metadata['train_filename']
      else:
         self.filename = metadata['test_filename']

   def read(self):
      if(self.data_type == DataType.txt):
         return self.__read_txt()
      return

   def __read_txt(self):
      file = open(self.path + self.filename,"r") 
      data_as_string = file.readlines()
      dataset = []
      for line in data_as_string:
         line_data = [float(i) for i in line.split()];
         dataset.append(line_data)

      dataset = np.array(dataset)
      # This is specific to 50words
      data = dataset[:, 1:]
      labels = dataset[:, 0, np.newaxis]
      return (data, labels)




      

