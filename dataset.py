# Import necessary modules

import os
import sys
import getpass
import argparse

import cv2
import numpy as np
import numpy.ma as ma
import pandas as pd
import random as rand
import matplotlib.pyplot as plt

# clean terminal in the beginning
username = getpass.getuser()
osName = os.name
if osName == 'posix':
    os.system('clear')
else:
    os.system('cls')


class visualization():

    def __init__(self, target_file):

        self.images = []
        self.labels = []

        self.dirPath = os.path.dirname(os.path.abspath('visualization.ipynb'))
        self.imgPath = os.path.join(self.dirPath, 'data', 'raw')
        self.labelPath = os.path.join(self.dirPath, 'data', 'annotations')

        #                 houses,     buildings,    garage
        self.classes = [     1,           2,           3     ]
        self.colors =  [(0, 0, 255), (0, 255, 0), (255, 0, 0)] #rgb

        # To processed scaled data as well
        self.scaled = True
        self.scaled_to = 50

        # target file 'test.txt' or 'train.txt'
        self.target_file = target_file

        # get list of images path
        with open(os.path.join(self.dirPath, self.target_file), 'r') as f:
            for line in f.readlines():

                # make lists of all the images path
                self.images.append(str(os.path.join(self.imgPath, line.split('.')[0]) + '.png'))

    def scale_image(self, image):

        #calculate the 50 percent of original dimensions
        width = int(image.shape[1] * self.scaled_to / 100)
        height = int(image.shape[0] * self.scaled_to / 100)

        # input image is too big, resize it
        dsize = (width, height)
        new_image = cv2.resize(image, dsize, interpolation = cv2.INTER_CUBIC)

        return new_image

    def draw_polygons(self, img_name, image, vertices, obj, bbox=True):

        # draw segmented polygon
        cv2.drawContours(image, [vertices], contourIdx= 0, color=self.colors[obj], thickness= -1)

        # draw rectangle
        if bbox:
            rect = cv2.minAreaRect(vertices)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(image,[box],0,(0,0,255),2)

        # visualize semantic segmentation
        cv2.imshow(img_name, image)

    def empty_mask(self, image, rescaled=True):

        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        if rescaled == True:
            mask = self.scale_image(mask)

        return mask

    def generate_dataset(self):

        houses = []
        buildings  = []
        garages = []

        try:
            # Main loop
            for img in self.images:

                # read image
                rgb = cv2.imread(img)

                # create empty mask
                mask = self.empty_mask(rgb, rescaled=True)

                if self.scaled == True:
                    scaled_rgb = self.scale_image(rgb)

                # read label file corresponding to each rgb image
                fileNo = (img.split('/')[-1].split('.')[0] + '.png-annotated.json')
                fileNo = os.path.join(self.labelPath, fileNo)
                data = pd.read_json(fileNo)

                # iterate through each structure
                for structure in range(len(data.labels)):
                    print('found {} {}'.format(len(data.labels[structure]['annotations']), data.labels[structure]['name']))

                    # create lists of class samples per image
                    if self.classes[0] == (structure+1):
                        houses.append(len(data.labels[structure]['annotations']))
                    elif self.classes[1] == (structure+1):
                        buildings.append(len(data.labels[structure]['annotations']))
                    else:
                        garages.append(len(data.labels[structure]['annotations']))

                    # iterate thorough each polygons
                    for premises in range(len(data.labels[structure]['annotations'])):

                        # get vertices of polygons (house, building, garage)
                        vertices = np.array(data.labels[structure]['annotations'][premises]['segmentation'], np.int32)
                        vertices = vertices.reshape((-1,1,2))

                        # draw polygons on original image to create segmentation
                        # self.draw_polygons('raw_rgb', rgb, vertices, structure, bbox=True)

                        scaled_vertices = []
                        if self.scaled == True:

                            # draw polygons on scaled image
                            for v in range(len(vertices)):
                                scaled_vertices.append(int(vertices[v][0][0] * self.scaled_to / 100)) #x
                                scaled_vertices.append(int(vertices[v][0][1] * self.scaled_to / 100)) #y
                            scaled_vertices = np.array(scaled_vertices).reshape((-1,1,2))

                            # draw polygons on scaled image to create segmentation
                            self.draw_polygons('scaled_rgb', scaled_rgb, scaled_vertices, structure, bbox=True)

                            # print(scaled_vertices)
                            # print(scaled_vertices[0][0])
                            # print(scaled_vertices[-1])
                            # print(zip(*scaled_vertices))

                            # print(scaled_rgb.shape)
                            # print(mask.shape)
                            mask = cv2.cvtColor(scaled_rgb, cv2.COLOR_BGR2GRAY)
                            res = cv2.bitwise_and(scaled_rgb, scaled_rgb, mask = mask)
                            # print(type(mask))
                            # mask = mask + scaled_rgb
                            cv2.imshow('mask', mask)


                # break with Esc
                key = cv2.waitKey(50000) & 0xFF
                if key == 27:
                    print('stopped!')
                    break
                elif key == ord('n'):
                    continue
        finally:
            # print listn
            print('houses:', houses)
            print('buildings:', buildings)
            print('garages:', garages)
            cv2.destroyAllWindows()


if __name__ == '__main__':

    # Get command line arguments
    args = argparse.ArgumentParser()
    args.add_argument('--targetFile', required=False, default= 'total.txt', help='select whether to visualize test.txt or train.txt dataset')
    opt = args.parse_args()
    print('visualizing {} dataset'.format(opt.targetFile))

    # visualize dataset
    viz = visualization(opt.targetFile)
    viz.generate_dataset()
