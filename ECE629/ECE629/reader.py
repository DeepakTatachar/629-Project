import numpy as np
from enum import Enum
import numpy as np
import scipy.io as sio

class DataType(Enum):
   # Enum for each type of supported data set
   txt = 1
   mat = 2

class reader(object):
   ''' Class that reads the dataset based on the type and returns the 
   read dataset and labels, takes the input whether the dataset to be read
   is a part of the testing or training set'''

   # Data type, two types, 'txt' for all the time series datasets
   # 'mat' for the financial data 
   data_type = DataType.txt
   
   # Holds the relative path of the dataset
   path = './'

   # Names of the training set and testing set
   train_filename = ''
   test_filename = ''

   # Holds the name of the current data set being read
   dataset_name = ''

   # Names of the labels, only relevant for 'mat' types
   # Not the best way to do this, but will do for the project at hand
   train_labels_filename = ''
   test_labels_filename = ''

   def __init__(self, parsedJSON, dataset_name):
      # metadata is read from the JSON file, so its a dictionary
      metadata = parsedJSON[dataset_name]
      self.dataset_name = dataset_name
      self.data_type = DataType[metadata['type']]
      self.path = metadata['path']
      self.train_filename = metadata['train_filename']
      self.test_filename = metadata['test_filename']

      # Hack to deal with mat types, idealy I would have an abstracted class
      # one for txt and one for mat types, I will do that if I have the time.
      # TODO: If time permits create a subclass, one for each type of dataset,
      # and restructure this class
      if(self.data_type == DataType.mat):
         self.test_labels_filename = metadata['test_labels']
         self.train_labels_filename = metadata['train_labels']
      else:
         self.train_labels_filename = ''
         self.test_labels_filename = ''

   def read(self, train=True):
      if(self.data_type == DataType.txt):
         return self.__read_txt(train)

      if(self.data_type == DataType.mat):
         return self.__read_mat(train)

      return

   def __read_mat(self, train=True):
      ''' Handles the reading of 'mat' type input data set
      Automatically called when the data set begin read is 'mat' type
      Called from read (public) this function is private'''
      if(train):
         filename = self.path + self.train_filename
         label = self.path + self.train_labels_filename
      else:
         filename = self.path + self.test_filename
         label = self.path + self.test_labels_filename

      data_dic = sio.loadmat(filename)
      labels_dic = sio.loadmat(label)

      data = data_dic[self.dataset_name]
      labels = labels_dic[self.dataset_name]

      return (data.transpose(), np.reshape(labels, labels.shape[1]))

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




      

