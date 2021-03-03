# Split dataset into testing & training sets # NOTE: do this one once

import os
# import pandas as pd
import random as rand

from utils import*

dirPath = os.path.dirname(os.path.abspath('visualization.ipynb'))
print('directory path:', dirPath)

# get all files name within the directory
files = os.listdir('data/annotations/')
# print(files[0])
# print(files[0].split('.')[0].split('_'))
# print(files)

#NOTE: shuffle and split based on ``stratified cross-validation``

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


# from skimage import io
# from skimage import color
# from skimage import segmentation
# import matplotlib.pyplot as plt


# # Load tiger image from URL
# satellite = io.imread(os.path.join(dirPath, 'data/raw', '0_0.png'))

# # Segment image with SLIC - Simple Linear Iterative Clustering
# seg = segmentation.slic(satellite, n_segments=3, compactness=40.0, enforce_connectivity=True, sigma=3)

# # Generate automatic colouring from classification labels
# io.imshow(color.label2rgb(seg,satellite))
# plt.show()