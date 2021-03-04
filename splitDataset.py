# Split dataset into testing & training sets # NOTE: do this one once

import os
# import pandas as pd
import random as rand
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from utils import*

dirPath = os.path.dirname(os.path.abspath('visualization.ipynb'))
# print('directory path:', dirPath)

# get all files name within the directory
files = os.listdir('data/annotations/')
# print(files[0])
# print(files[0].split('.')[0].split('_'))
# print(files)

# shuffle input dataset
total_samples = len(files)
shuffle_indices = rand.sample(files, total_samples)

# erase old files
trainTxt = os.path.join(dirPath, 'train.txt')
testTxt = os.path.join(dirPath, 'test.txt')
total = os.path.join(dirPath, 'total.txt')
open(trainTxt, 'w').close()
open(testTxt, 'w').close()
open(total, 'w').close()




# split shuffled dataset into training(80%) and testing(20%)
print('\ncreating train and test sets with random order 80-20%')
create_sets(trainTxt, 0, int(total_samples*0.80), shuffle_indices) # train samples
create_sets(testTxt, int(total_samples*0.80), total_samples, shuffle_indices) # test samples
create_sets(total, 0, total_samples, shuffle_indices) # total samples




# #NOTE: shuffle and split based on ``stratified cross-validation``
# houses = [5, 101, 138, 33, 49, 57, 100, 52, 82, 146, 229, 217, 116, 41, 83, 61, 73, 77, 5, 121, 183, 56, 213, 87, 60, 36, 26, 99, 127, 26, 20, 320, 72, 3, 176, 89, 153, 110, 143, 17, 59, 115, 338, 268, 106, 165, 123, 84, 32, 11, 8, 83, 91, 41, 110, 113, 44, 59, 107, 62, 16, 99, 240, 89, 51, 66, 65, 65, 65, 6, 81, 125]
# buildings = [43, 167, 201, 172, 194, 113, 148, 138, 125, 190, 120, 109, 153, 73, 225, 130, 111, 129, 22, 124, 171, 239, 99, 50, 195, 102, 216, 275, 156, 51, 94, 137, 169, 13, 158, 87, 92, 145, 150, 74, 59, 123, 86, 292, 205, 243, 67, 90, 152, 17, 111, 156, 167, 142, 93, 89, 69, 154, 67, 199, 137, 207, 187, 150, 88, 113, 161, 92, 131, 30, 127, 39]
# garages = [3, 7, 9, 9, 1, 1, 5, 2, 0, 2, 9, 7, 3, 10, 5, 0, 0, 0, 1, 12, 9, 5, 0, 5, 2, 4, 1, 4, 2, 1, 1, 15, 1, 0, 0, 9, 6, 8, 10, 4, 2, 12, 3, 0, 0, 3, 6, 25, 1, 0, 7, 0, 0, 7, 0, 1, 0, 2, 7, 0, 2, 0, 1, 12, 0, 3, 3, 7, 7, 8, 1, 8]

# # print(np.array([[houses], [buildings], [garages]]))
# # print(np.array([[houses], [buildings], [garages]]).shape)

# # print(np.array([houses, buildings, garages]))
# # print(np.array([houses, buildings, garages]).shape)
# X = np.array([houses, buildings, garages])
# # print(X[0,:])

# y = np.zeros((3,72))
# y[0,:] = 1
# y[1,:] = 2
# y[2,:] = 3
