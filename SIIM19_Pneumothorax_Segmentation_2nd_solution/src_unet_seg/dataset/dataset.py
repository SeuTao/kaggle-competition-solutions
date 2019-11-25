from torchvision import transforms
import torchvision.datasets as datasets
from PIL import Image
import torch.utils.data as data
import torch
import numpy as np
import cv2
from tqdm import tqdm
import random
import albumentations
# from tuils.preprocess import *
from torch.utils.data.dataloader import default_collate
from os.path import isfile
from skimage import feature
from scipy import ndimage
import json

MAX_SIZE = 448
IMAGENET_SIZE = 416
RESIZE_SIZE = 1024

fpath = open('../configs/seg_path_configs.json', encoding='utf-8')
path_data = json.load(fpath)
train_img_path = path_data['train_img_path']
test_img_path = path_data['test_img_path']
extra_img_path = path_data['extra_img_path']


def expand_path(p):
    p = str(p)
    if isfile(train_img_path + p):
        return train_img_path + p
    if isfile(test_img_path + p):
        return test_img_path + p
    if isfile(extra_img_path + p):
        return extra_img_path + p
    else:
        print(p)
        print('ERROR : no image file in {}'.format(p))
        return p


def rle2mask(rle, width, height):
    mask= np.zeros(width * height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height)


class Siim_Dataset(data.Dataset):

    def __init__(self,
                 df = None,
                 name_list = None,
                 transform = None
                 ):
        self.df = df
        self.name_list = name_list
        self.transform = transform

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):

        name = self.name_list[idx]
        image_path = expand_path(name)
        # image = cv2.imread('/home/vip415/HJH/mmdetectionV2/siim_tool/SIIM/train_png/' + name)
        image = cv2.imread(image_path)
        rle = self.df[self.df['ImageId'] == (name.replace('.png', ''))]['EncodedPixels']
        if rle.values[0] == ' -1':
            masks = np.zeros((1024, 1024))
        else:
            masks = [np.expand_dims(rle2mask(x, 1024, 1024).T,axis=0) for x in rle]
            masks = np.sum(masks,0)
            masks[masks > 1] = 1
            masks = masks[0, :, :]

        if self.transform is not None:
            augmented = self.transform(image=image, mask=masks)
            image = augmented['image'].transpose(2, 0, 1)
            masks = np.expand_dims(augmented['mask'], axis=0)

        return image, masks


class Siim_Dataset_cls_seg_train(data.Dataset):

    def __init__(self,
                 df = None,
                 name_list = None,
                 transform = None
                 ):
        self.df = df
        self.name_list = name_list
        self.transform = transform
        # self.kernel = np.ones((5,5), np.uint8)

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):

        name = self.name_list[idx]
        image_path = expand_path(name)
        image = cv2.imread(image_path)
        if name.startswith('patient'):
            rle = self.df[self.df['ImageId'] == (name.replace('.jpg', ''))]['EncodedPixels']
        else:
            rle = self.df[self.df['ImageId'] == (name.replace('.png', ''))]['EncodedPixels']
        if rle.values[0] == '-1':
            masks = np.zeros((1024, 1024))
            cls_label = torch.FloatTensor([0])
        elif rle.values[0] == '2':
            masks = np.zeros((1024, 1024))
            cls_label = torch.FloatTensor([1])            
        else:
            # print(rle)
            masks = [np.expand_dims(rle2mask(x, 1024, 1024).T,axis=0) for x in rle]
            masks = np.sum(masks,0)
            masks[masks>1] = 1
            masks = masks[0, :, :]
            # masks = cv2.dilate(masks, self.kernel, iterations=2)
            cls_label = torch.FloatTensor([1])

        if self.transform is not None:
            augmented = self.transform(image=image, mask=masks)
            image = augmented['image'].transpose(2, 0, 1)
            masks = np.expand_dims(augmented['mask'], axis=0)

        return image, masks, cls_label


class Siim_Dataset_cls_seg_val(data.Dataset):

    def __init__(self,
                 df=None,
                 name_list = None,
                 transform = None
                 ):
        self.df = df
        self.name_list = name_list
        self.transform = transform

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):

        name = self.name_list[idx]
        image_path = expand_path(name)
        image = cv2.imread(image_path)
        rle = self.df[self.df['ImageId']==(name.replace('.png', ''))]['EncodedPixels']
        # print(rle.values[0])
        if rle.values[0] == '-1':
            masks = np.zeros((1024, 1024))
            cls_label = torch.FloatTensor([0])
        elif rle.values[0] == '2':
            masks = np.zeros((1024, 1024))
            cls_label = torch.FloatTensor([1])            
        else:
            # print(rle)
            masks = [np.expand_dims(rle2mask(x, 1024, 1024).T,axis=0) for x in rle]
            masks = np.sum(masks,0)
            masks[masks>1] = 1
            masks = masks[0, :, :]
            cls_label = torch.FloatTensor([1])

        if self.transform is not None:
            augmented = self.transform(image=image, mask=masks)
            image = augmented['image'].transpose(2, 0, 1)
            masks = np.expand_dims(augmented['mask'], axis=0)

        return image, masks, cls_label


def mask_to_instances(masks):
    for i in range(30-5):
        instance = (masks & (2 ** i))
        if instance.any():
            yield instance


def generate_dataset_loader(df_all, c_train, train_transform, train_batch_size, c_val, val_transform, val_batch_size, workers):

    train_dataset = Siim_Dataset(df_all, c_train, train_transform)
    val_dataset = Siim_Dataset(df_all, c_val, val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,        
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        # sampler=train_sampler,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batch_size,        
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        # sampler=val_sampler,
        drop_last=False)

    return train_loader, val_loader


def generate_dataset_loader_cls_seg(df_all, c_train, train_transform, train_batch_size, c_val, val_transform, val_batch_size, workers):

    train_dataset = Siim_Dataset_cls_seg_train(df_all, c_train, train_transform)
    val_dataset = Siim_Dataset_cls_seg_val(df_all, c_val, val_transform)

    # train_dataset = Siim_Dataset_multi_task_train(df_all, c_train, train_transform)
    # val_dataset = Siim_Dataset_multi_task_val(df_all, c_val, val_transform)

    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,        
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        # sampler=train_sampler,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batch_size,        
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        # sampler=val_sampler,
        drop_last=False)

    return train_loader, val_loader