#! /anaconda3/envs/yolact/bin/Python3

#%%
# https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb

import os
import sys
import getpass
import argparse

import cv2
import numpy as np
import numpy.ma as ma
import pandas as pd
import random as rand
import albumentations as albu
import matplotlib.pyplot as plt
from IPython.display import clear_output

from sklearn.model_selection import train_test_split

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

import segmentation_models_pytorch as smp


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
CLASSES = ['houses', 'buildings', 'garages'] #1,2,3


# Visualize preprocessed images
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

# from preprocess import*
# data = preprocess('total.txt')

# data path to raw image (random index)
MAIN_DIR = os.path.dirname(os.path.join(os.path.abspath('data'),'data'))
RGB_DIR = os.path.join(MAIN_DIR, 'rgb')
MASK_DIR = RGB_DIR.replace('rgb','mask')
# print('MAIN_DIR:', MAIN_DIR)
# print('RGB_DIR:', RGB_DIR)
# print('MASK_DIR:', MASK_DIR)


# all sample indices
frames_idx = os.listdir(RGB_DIR)

# create training and testing sets
train_idx, valid_idx = train_test_split(frames_idx, test_size = 0.30)
eval_idx = os.listdir(os.path.join(MAIN_DIR, 'eval'))
# print(train_idx)
# print(eval_idx)
# print(valid_idx)


""" visualize random rgb, mask images """
# # read rgb
# rgbPath = os.path.join(RGB_DIR,frames_idx[rand.choice(range(0,72))])
# rgb = cv2.imread(rgbPath)
# print(rgbPath)
# # print(rgb.shape)

# # read mask
# maskPath = rgbPath.replace('rgb', 'mask')
# mask = cv2.imread(maskPath, 0)
# print(maskPath)
# # print(mask.shape)

# # read bbox
# bboxPath = rgbPath.replace('rgb', 'bbox').replace('png', 'txt')
# print(bboxPath)

# #NOTE: RESCALE POLYGONS AS PER IMAGE SIZE
# # read polygons
# labelPath = rgbPath.replace('rgb', 'annotations').replace('png', 'png-annotated.json')
# print(labelPath)

# # # scale_image = data.scale_image(rgb)
# # # height, width, channel = scale_image.shape

# visualize(image=rgb, satellite_mask=mask.squeeze())



class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    classes = CLASSES

    def __init__(self, images_dir, masks_dir, set_indices, classes=None, augmentation=None, preprocessing=None):

        self.ids = set_indices
        # print(len(self.ids))
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.classes.index(cls.lower())+1 for cls in classes]
        # print(self.class_values)

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        # mask = plt.imread(self.masks_fps[i])

        # print('before rgb shape', image.shape)
        # print('before mask shape', mask.shape)

        # # resize images
        dsize = (640, 480)
        image =  cv2.resize(image, dsize, interpolation = cv2.INTER_CUBIC)
        mask =  cv2.resize(mask, dsize, interpolation = cv2.INTER_CUBIC)

        # # print('rgb shape', image.shape)
        # # print('mask shape', mask.shape)

        # extract certain classes from mask (e.g. cars)
        # NOTE assign class values to mask NOTE
        # print('before\n',mask.shape)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float') #channels == totalClasses
        # mask = mask.squeeze()
        # print('after\n',mask.shape)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)


# sample trainDataset
dataset = Dataset(RGB_DIR, MASK_DIR, train_idx, classes=CLASSES)

# get some sample
image, mask = dataset[3]

# visualize
visualize(
        satellite=image,
        houses=mask[...,0].squeeze(),
        buildings=mask[...,1].squeeze(),
        garages=mask[...,2].squeeze()),

# print(mask[...,2].squeeze())

# try:
#     cv2.imshow('mask', mask)
#     cv2.waitKey(5000)
# finally:
    # cv2.destroyAllWindows()

#%%

# ### Augmentations

def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        # 768 because our scaled images is (864,785): and ~768//32
        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        albu.RandomCrop(height=320, width=320, always_apply=True),

        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        # albu.PadIfNeeded(384, 480)
        # albu.PadIfNeeded(480, 384)
        albu.PadIfNeeded(480, 320)
    ]
    return albu.Compose(test_transform)


# augmented_dataset = Dataset(
#     RGB_DIR,
#     MASK_DIR,
#     train_idx,
#     augmentation=get_training_augmentation(),
#     classes=CLASSES
# )

# # random augmented random transforms samples
# for i in range(3):
#     image, mask = augmented_dataset[1]
#     visualize(image=image, mask=mask.squeeze())


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)



#%%

# Create model

# # create segmentation model with pretrained encoder
ACTIVATION =  'softmax2d'  #'sigmoid' # could be None for logits or 'softmax2d' for multicalss segmentation
DEVICE = 'cuda'


# ENCODER = 'se_resnext50_32x4d'
# ENCODER_WEIGHTS = 'imagenet'

# model = smp.FPN(
#     encoder_name=ENCODER,
#     encoder_weights=ENCODER_WEIGHTS,
#     classes=len(CLASSES),
#     activation=ACTIVATION,
# )


ENCODER = 'resnet18'
# ENCODER_WEIGHTS = 'imagenet'
ENCODER_WEIGHTS = 'swsl'

# aux_params=dict(
#     pooling='avg',             # one of 'avg', 'max'
#     dropout=0.5,               # dropout ratio, default is None
#     activation='sigmoid',      # activation function, default is None
#     classes=3,                 # define number of output labels
# )

model = smp.Unet(
        encoder_name=ENCODER,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=ENCODER_WEIGHTS,     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        # aux_params=aux_params,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


train_dataset = Dataset(
    RGB_DIR,
    MASK_DIR,
    train_idx,
    # augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

valid_dataset = Dataset(
    RGB_DIR,
    MASK_DIR,
    valid_idx,
    # augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=12)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

# Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
# IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=0.001),
])

# create epoch runners
# it is a simple loop of iterating over dataloader`s samples
train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)


#%%
# train model for 40 epochs

max_score = 0

for i in range(0, 200):

    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)

    # do something (save model, change lr, etc.)
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, './best_model.pth')
        print('Model saved!')

    if i == 50:
        optimizer.param_groups[0]['lr'] = 1e-2
        print('Decrease decoder learning rate to 1e-2!')
    elif i == 150:
        optimizer.param_groups[0]['lr'] = 1e-4
        print('Decrease decoder learning rate to 1e-4!')



#%%


# create test dataset
test_dataset = Dataset(
    RGB_DIR,
    MASK_DIR,
    valid_idx,
    # augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

best_model = torch.load('./best_model.pth')
test_dataloader = DataLoader(test_dataset)

# metrics = [
#     smp.utils.metrics.IoU(threshold=0.2),
# ]


# evaluate model on test set
test_epoch = smp.utils.train.ValidEpoch(
    model=best_model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
)

logs = test_epoch.run(test_dataloader)

#%%

# test dataset without transformations for image visualization
test_dataset_vis = Dataset(
    RGB_DIR,
    MASK_DIR,
    valid_idx,
    classes=CLASSES,
)

for i in range(5):
    n = np.random.choice(len(test_dataset))

    image_vis = test_dataset_vis[n][0].astype('uint8')
    image, gt_mask = test_dataset[n]

    gt_mask = gt_mask.squeeze()

    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())


    # dsize = (480, 320)
    # image_vis =  cv2.resize(image, dsize, interpolation = cv2.INTER_CUBIC)

    visualize(
        image_vis=image_vis,
        # image=np.transpose(image, (1, 2, 0)),
        ground_truth_mask=np.transpose(gt_mask, (1, 2, 0)),
        predicted_mask= pr_mask, #np.transpose(pr_mask, (1, 2, 0))
    )

# try:
#     cv2.imshow('mask', cv2.cvtColor(np.transpose(pr_mask, (1, 2, 0)), cv2.COLOR_RGB2GRAY ))
#     cv2.waitKey(1000)
# finally:
#     cv2.destroyAllWindows()


print(image_vis.shape)
print(image.shape)
print(gt_mask.shape)
print(pr_mask.shape)

