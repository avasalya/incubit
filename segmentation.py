#! /anaconda3/envs/yolact/bin/Python3

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
CLASSES = ['background', 'houses', 'buildings', 'garages'] #0,1,2,3
# CLASSES = ['houses', 'buildings', 'garages'] #1,2,3


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
MASK_DIR = RGB_DIR.replace('rgb','mask_bb')
# print('MAIN_DIR:', MAIN_DIR)
# print('RGB_DIR:', RGB_DIR)
# print('MASK_DIR:', MASK_DIR)


# all sample indices
frames_idx = os.listdir(RGB_DIR)

# create training and testing sets
train_idx, valid_idx = train_test_split(frames_idx, test_size = 0.25)
eval_idx = os.listdir(os.path.join(MAIN_DIR, 'eval'))
# print(train_idx)
# print(valid_idx)
# print(eval_idx)


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
        # self.class_values = [self.classes.index(cls.lower())+1 for cls in classes]
        self.class_values = [self.classes.index(cls.lower()) for cls in classes]#with bg as class
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
        dsize = (640, 640)
        image =  cv2.resize(image, dsize, interpolation = cv2.INTER_CUBIC)
        mask =  cv2.resize(mask, dsize, interpolation = cv2.INTER_CUBIC)

        # # print('rgb shape', image.shape)
        # # print('mask shape', mask.shape)

        # extract certain classes from mask
        # print('before\n',mask.shape)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float') #channels == totalClasses
        # mask = mask.squeeze()
        # print('after\n',mask.shape)

        # # add background if mask is not binary
        # if mask.shape[-1] != 1:
        #     background = 1 - mask.sum(axis=-1, keepdims=True)
        #     mask = np.concatenate((mask, background), axis=-1)

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
# dataset = Dataset(RGB_DIR, MASK_DIR, train_idx, classes=CLASSES)

# get some sample
# image, mask = dataset[1]

# # visualize
# visualize(
#         satellite=image,
#         bg=mask[...,0].squeeze(),
#         houses=mask[...,1].squeeze(),
#         buildings=mask[...,2].squeeze(),
#         garages=mask[...,3].squeeze()),

# print(mask[...,1].squeeze())

# try:
#     cv2.imshow('mask', mask)
#     cv2.waitKey(5000)
# finally:
    # cv2.destroyAllWindows()


# ### Augmentations

def get_training_augmentation():

    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        # albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        albu.PadIfNeeded(min_height=640, min_width=640, always_apply=False, border_mode=0),
        # albu.RandomCrop(height=320, width=320, always_apply=True),

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
        # albu.PadIfNeeded(384, 640)
        albu.PadIfNeeded(640, 640)
    ]
    return albu.Compose(test_transform)

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



augmented_dataset = Dataset(
    RGB_DIR,
    MASK_DIR,
    train_idx,
    augmentation=get_training_augmentation(),
    classes=CLASSES
)

# random augmented random transforms samples
for i in range(1):
    image, mask = augmented_dataset[1]
    # visualize(image=image, mask=mask.squeeze())
    # visualize
    visualize(
            satellite=image,
            bg=mask[...,0].squeeze(),
            houses=mask[...,1].squeeze(),
            buildings=mask[...,2].squeeze(),
            garages=mask[...,3].squeeze()),



# Create model

# # create segmentation model with pretrained encoder
DEVICE = 'cuda'

# could be None for logits or 'softmax2d' for multicalss segmentation
# ACTIVATION = 'sigmoid'
ACTIVATION =  'softmax2d'


# ENCODER = 'se_resnext50_32x4d'
# ENCODER = 'resnet50'
# ENCODER = 'resnet18'
# ENCODER = 'densenet161'
ENCODER = 'efficientnet-b4'

ENCODER_WEIGHTS = 'imagenet'
# ENCODER_WEIGHTS = 'swsl'

n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation

model = smp.FPN(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    activation=ACTIVATION,
)

# aux_params=dict(
#     pooling='avg',             # one of 'avg', 'max'
#     dropout=0.5,               # dropout ratio, default is None
#     activation='sigmoid',      # activation function, default is None
#     classes=3,                 # define number of output labels
# )

# model = smp.Unet(
#         encoder_name=ENCODER,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#         encoder_weights=ENCODER_WEIGHTS,  # use `imagenet` pre-trained weights for encoder initialization
#         in_channels=len(CLASSES),       # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#         activation= ACTIVATION,
#         # aux_params=aux_params,
# )


if torch.cuda.is_available():
    print('Cuda available')
    model.cuda()


preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


train_dataset = Dataset(
    RGB_DIR,
    MASK_DIR,
    train_idx,
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=12)


valid_dataset = Dataset(
    RGB_DIR,
    MASK_DIR,
    valid_idx,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)


loss = smp.utils.losses.DiceLoss()
#loss = smp.utils.losses.CrossEntropyLoss()

metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
    # smp.utils.metrics.Accuracy()
    #smp.utils.metrics.Fscore()
]

optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=0.0001),
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



# train model

max_score = 0
max_epochs = 200

#train accurascy, train loss, val_accuracy, val_loss
x_epoch_data = []
train_dice_loss = []
valid_dice_loss = []
train_acc_score = []
valid_acc_score = []


start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()

for i in range(0, max_epochs):

    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)

    x_epoch_data.append(i)
    train_dice_loss.append(train_logs['dice_loss'])
    valid_dice_loss.append(valid_logs['dice_loss'])

    train_acc_score.append(train_logs['iou_score'])
    valid_acc_score.append(valid_logs['iou_score'])

    # train_acc_score.append(train_logs['accuracy'])
    # valid_acc_score.append(valid_logs['accuracy'])

    # do something (save model, change lr, etc.)

    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, './best_model.pth')
        print('Model saved!')


    # if max_score < valid_logs['accuracy']:
    #     max_score = valid_logs['accuracy']
    #     torch.save(model, './best_model.pth')
    #     print('Model saved!')


    # if i == 25:
    #     optimizer.param_groups[0]['lr'] = 1e-5
    #     print('Decrease decoder learning rate to 1e-5!')

    if i == 150:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 5e-6!')

    if i == 175:
        optimizer.param_groups[0]['lr'] = 1e-6
        print('Decrease decoder learning rate to 1e-6!')

    # if i == 50:
    #     optimizer.param_groups[0]['lr'] = 5e-2
    #     print('Decrease decoder learning rate to 1e-5!')
    # elif i == 100:
    #     optimizer.param_groups[0]['lr'] = 1e-2
    #     print('Decrease decoder learning rate to 1e-6!')
    # elif i == 400:
    #     optimizer.param_groups[0]['lr'] = 1e-7
    #     print('Decrease decoder learning rate to 1e-7!')

end.record()

# Waits for everything to finish running
torch.cuda.synchronize()

time = start.elapsed_time(end)
print("Time elapsed: " + str(start.elapsed_time(end) / 60000) + " minutes")  # millisecond

# plot loss

fig = plt.figure(figsize=(14, 5))

ax1 = fig.add_subplot(1, 2, 1)
line1, = ax1.plot(x_epoch_data,train_dice_loss,label='train')
line2, = ax1.plot(x_epoch_data,valid_dice_loss,label='validation')
ax1.set_title("dice loss")
ax1.set_xlabel('epoch')
ax1.set_ylabel('dice_loss')
ax1.legend(loc='upper right')

ax2 = fig.add_subplot(1, 2, 2)
line1, = ax2.plot(x_epoch_data,train_acc_score,label='train')
line2, = ax2.plot(x_epoch_data,valid_acc_score,label='validation')
ax2.set_title("iou_score")
ax2.set_xlabel('epoch')
ax2.set_ylabel('iou_score')
ax2.legend(loc='upper left')

plt.show()

#%% validate results

best_model = torch.load('./best_model.pth')

# create test dataset
test_dataset = Dataset(
    RGB_DIR,
    MASK_DIR,
    valid_idx,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

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
    gt_mask = np.transpose(gt_mask, (1, 2, 0))
    pr_mask = np.transpose(pr_mask, (1, 2, 0))

    # visualize(
    #     image_vis=image_vis,
    #     # image=np.transpose(image, (1, 2, 0)),
    #     ground_truth_mask= gt_mask,
    #     # predicted_mask= pr_mask,
    #     # predicted_mask1= pr_mask[...,0],
    #     predicted_mask2= pr_mask[...,1],
    #     predicted_mask3= pr_mask[...,2],
    #     predicted_mask4= pr_mask[...,3],
    # )


    gt_mask_gray = np.zeros((gt_mask.shape[0],gt_mask.shape[1]))

    for ii in range(gt_mask.shape[2]):
        gt_mask_gray = gt_mask_gray + 1/gt_mask.shape[2]*ii*gt_mask[:,:,ii]

    pr_mask_gray = np.zeros((pr_mask.shape[0],pr_mask.shape[1]))
    for ii in range(pr_mask.shape[2]):
        pr_mask_gray = pr_mask_gray + 1/pr_mask.shape[2]*ii*pr_mask[:,:,ii]

    visualize(
        image=image_vis,
        ground_truth_mask=gt_mask[...,1:],
        predicted_mask=pr_mask_gray,
        # predicted_mask=pr_mask[1].squeeze()
    )




# try:
#     # cv2.imshow('mask', cv2.cvtColor(np.transpose(pr_mask, (1, 2, 0)), cv2.COLOR_RGB2GRAY ))
#     cv2.imshow('mask', pr_mask)
#     cv2.waitKey(10000)
# finally:
#     cv2.destroyAllWindows()


print('image_vis', image_vis.shape)
print('gt_mask', gt_mask.shape)
print('pr_mask', pr_mask.shape)

#%%
# find contours
# imgray = cv2.cvtColor(pr_mask[:,:,1], cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(pr_mask[...,1], 127, 255, 0)

# try:
#     cv2.imshow('mask',pr_mask)
#     cv2.waitKey(10000)
# finally:
#     cv2.destroyAllWindows()


# im, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)










# # https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb