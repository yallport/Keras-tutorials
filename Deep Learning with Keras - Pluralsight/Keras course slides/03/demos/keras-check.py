#   keras-check.py
#   Verify that Keras can interact with the backend

import numpy as np
from keras import backend as kbe

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

# Test Keras - backend interaction
data = kbe.variable(np.random.random((4,2)))   # create 4 X 2 tensor of random numbers 
zero_data = kbe.zeros_like(data)               # create 4 X 2 tensor of zeros
print(kbe.eval(zero_data))                     # evaluate the zero_data and print out the results