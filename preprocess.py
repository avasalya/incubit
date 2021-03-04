# Import necessary modules
#%%
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
from IPython.display import clear_output


# clean terminal in the beginning
username = getpass.getuser()
osName = os.name
if osName == 'posix':
    os.system('clear')
else:
    os.system('cls')
class preprocess():

    def __init__(self, target_file):

        self.images = []
        self.labels = []

        self.dirPath = os.path.dirname(os.path.abspath(__file__))
        self.imgPath = os.path.join(self.dirPath, 'data', 'raw')
        self.maskPath = os.path.join(self.dirPath, 'data', 'mask')
        self.bboxPath = os.path.join(self.dirPath, 'data', 'bbox')
        self.labelPath = os.path.join(self.dirPath, 'data', 'annotations')

        #                 houses,     buildings,    garage
        self.classes = [     1,           2,           3     ]
        self.colors =  [(0, 0, 255), (0, 255, 0), (255, 0, 0)] #rgb

        # To processed scaled data as well
        self.scaled = True
        self.scaled_to = 50

        # target file 'test.txt' or 'train.txt'
        self.target_file = target_file

        #generate mask images
        self.show_mask = True
        self.write_img = False

        #generate bbox labels
        self.write_bbox = False

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

    def draw_polygons(self, img_name, image, vertices, obj, draw_bbox=True):

        # draw segmented polygon
        cv2.drawContours(image, [vertices], contourIdx= 0, color=self.colors[obj], thickness= -1)

        # draw (rotated) rectangle
        rect = cv2.minAreaRect(vertices)
        bbox = cv2.boxPoints(rect)
        bbox = np.int0(bbox)

        top_left = np.min(bbox, axis=0)
        bottom_right = np.max(bbox, axis=0)

        if draw_bbox:
            cv2.drawContours(image,[bbox],0,(255,100,100),2)

        # visualize semantic segmentation
        cv2.imshow(img_name, image)

        return np.append(top_left, bottom_right)#x1,y1,x2,y2

    def create_mask(self, image, rescaled=True):

        # create empty mask
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        if rescaled == True:
            mask = self.scale_image(mask)

        # create binary mask for each class
        mask_r = cv2.inRange(image, (0,0,255), (0,0,255))#house
        mask_g = cv2.inRange(image, (0,255,0), (0,255,0))#building
        mask_b = cv2.inRange(image, (255,0,0), (255,0,0))#garage

        # create 1 channel mask with r,b,g channels
        img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mask_r = cv2.bitwise_and(img, img, mask = mask_r)
        mask_g = cv2.bitwise_and(img, img, mask = mask_g)
        mask_b = cv2.bitwise_and(img, img, mask = mask_b)

        # merge all classes mask
        mask = np.stack((mask_b + mask_g + mask_r), axis=0)
        # print(mask.shape)

        # create mask with r,b,g channels
        mask_r = cv2.bitwise_and(image, image, mask = mask_r)
        mask_g = cv2.bitwise_and(image, image, mask = mask_g)
        mask_b = cv2.bitwise_and(image, image, mask = mask_b)
        rgb_mask = np.stack((mask_b + mask_g + mask_r), axis=0)

        print(mask.shape)
        return mask, rgb_mask

    def generate_dataset(self):

        houses = []
        buildings  = []
        garages = []

        try:
            # Main loop
            for img in self.images:

                # read image
                rgb = cv2.imread(img)

                # rescaled raw image
                if self.scaled == True:
                    scaled_rgb = self.scale_image(rgb)
                    # save rescaled rgb image
                    # if self.write_img == True:
                        # cv2.imwrite(img.replace('raw', 'rgb'), scaled_rgb) #rgb

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
                        # bbox = self.draw_polygons('raw_rgb', rgb, vertices, structure, bbox=True)

                        scaled_vertices = []
                        scaled_bboxes  = []
                        if self.scaled == True:

                            # draw polygons on scaled image
                            for v in range(len(vertices)):
                                scaled_vertices.append(int(vertices[v][0][0] * self.scaled_to / 100)) #x
                                scaled_vertices.append(int(vertices[v][0][1] * self.scaled_to / 100)) #y
                            scaled_vertices = np.array(scaled_vertices).reshape((-1,1,2))

                            # draw polygons on scaled image to create segmentation
                            scaled_bbox = self.draw_polygons('scaled_rgb', scaled_rgb, scaled_vertices, structure, draw_bbox=False)
                            scaled_bboxes.append(scaled_bbox)

                            # create segmentation mask
                            if self.show_mask == True:
                                scaled_mask, rgb_mask = self.create_mask(scaled_rgb, rescaled=True)
                                cv2.imshow('scaled_mask', scaled_mask)
                                # plt.imshow(scaled_mask)

                                # normalized mask
                                # scaled_mask = np.asarray(scaled_mask)
                                # scaled_mask = scaled_mask/255
                # plt.show()

                # save bounding boxes
                if self.write_bbox == True:
                    with open(img.replace('raw', 'bbox').replace('png', 'txt'), 'w') as f:
                        for b in scaled_bboxes:
                            f.write(str(b))
                        f.write('\n')

                # same mask images #binary mask
                if self.write_img == True and self.show_mask == True:
                    # plt.imsave(img.replace('raw', 'mask'), scaled_mask)
                    # cv2.imwrite(img.replace('raw', 'rgbmask'), rgb_mask)
                    cv2.imwrite(img.replace('raw', 'graymask'), scaled_mask)

                # break with Esc, n to proceed next
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    print('stopped!')
                    break
                elif key == ord('n'):
                    continue
        finally:
            # print lists of samples
            print('houses:', houses)
            print('buildings:', buildings)
            print('garages:', garages)
            cv2.destroyAllWindows()

#%%
if __name__ == '__main__':

    # Get command line arguments
    args = argparse.ArgumentParser()
    args.add_argument('--targetFile', required=False, default= 'total.txt', help='select whether to visualize test.txt or train.txt dataset')
    opt = args.parse_args()
    print('visualizing {} dataset'.format(opt.targetFile))

    # visualize preprocess
    data = preprocess(opt.targetFile)
    #data = preprocess('total.txt')
    data.generate_dataset()
