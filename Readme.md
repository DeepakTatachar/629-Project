# ECE 629 Project Fall 2018
---------------------------
**Author: Deepak Ravikumar**\
**Purdue University**

Request: Please use this code in a fair manner.

To run the code the following dependencies need to be installed

•	keras\
•	numpy\
•	tensorflow\
•	scikit-image\
•	pyts\
•	scipy

The code has two major classes, the timeseries class and the reader class. 

The timeseries class holds the data, testing images, training images, number of classes, time series converted to GAF and MTF values. It has one major function, to read the input data set and convert it into the right format for training and testing. This is achieved through three functions of the class

•	readdataset\
•	convert_to_GASF\
•	convert_to_MTF

The other important object is reader, this is a class that reads the dataset based on the type and returns the read dataset and labels, takes the input whether the dataset to be read is a part of the testing or training set. This is done through JSON file, which is parsed to figure out the details of the dataset for example, the type of the dataset, number of classes and so on, the json file is ‘dict.json’.

To run 2-D converted images run main.py, to run the one-dim net run oneDim.py. To run on different datasets change the value of the parameter to time_series.readdataset() function, the value of the argument is the name of the dataset, to find the supported datasets look at the ‘dict.json’ file
