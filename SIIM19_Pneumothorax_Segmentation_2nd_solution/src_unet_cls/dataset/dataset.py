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
import json
# from tuils.preprocess import *
from torch.utils.data.dataloader import default_collate

fpath = open('../configs/seg_path_configs.json', encoding='utf-8')
path_data = json.load(fpath)
train_img_path = path_data['train_img_path']

def generate_transforms(image_size):
    # MAX_SIZE = 448
    IMAGENET_SIZE = image_size

    train_transform = albumentations.Compose([
            
        albumentations.Resize(IMAGENET_SIZE, IMAGENET_SIZE),
        albumentations.OneOf([
            albumentations.RandomGamma(gamma_limit=(60, 120), p=0.9),
            albumentations.RandomBrightness(limit=0.2, p=0.9),
            albumentations.RandomContrast(limit=0.2, p=0.9),
            albumentations.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=0.9),
            ]),
        albumentations.OneOf([
            albumentations.Blur(blur_limit=4, p=1),
            albumentations.MotionBlur(blur_limit=4, p=1),
            albumentations.MedianBlur(blur_limit=4, p=1)
            ], p=0.5),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, interpolation=cv2.INTER_LINEAR,border_mode=cv2.BORDER_CONSTANT, p=1),
        albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)

    ])

    val_transform = albumentations.Compose([
        albumentations.Resize(IMAGENET_SIZE, IMAGENET_SIZE),
        albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
    ])

    return train_transform, val_transform

def rle2mask(rle, width, height):
    mask= np.zeros(width* height)
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
        image = cv2.imread(train_img_path + name)
        rle = self.df[self.df['ImageId']==(name.replace('.png', '').replace('.jpg', ''))]['EncodedPixels']
        if rle.values[0] == ' -1':
            masks = np.zeros((1024, 1024))
        else:
            masks = [np.expand_dims(rle2mask(x, 1024, 1024).T,axis=0) for x in rle]
            masks = np.sum(masks,0)
            masks[masks>1] = 1
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
        self.kernel = np.ones((3,3), np.uint8)

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):

        name = self.name_list[idx]
        image = cv2.imread(train_img_path + name)
        rle = self.df[self.df['ImageId']==(name.replace('.png', '').replace('.jpg', ''))]['EncodedPixels']

        if rle.values[0] == '-1':
            masks = np.zeros((1024, 1024))
            cls_label = torch.FloatTensor([0])
        
        elif rle.values[0] == '2':
            masks = np.zeros((1024, 1024))
            cls_label = torch.FloatTensor([1])            
        else:

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


class Siim_Dataset_cls_seg_val(data.Dataset):

    def __init__(self,
                 df = None,
                 name_list = None,
                 transform = None
                 ):
        self.df = df
        self.name_list = name_list
        self.transform = transform
        self.kernel = np.ones((3,3), np.uint8) 

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):

        name = self.name_list[idx]
        image = cv2.imread(train_img_path + name)
        rle = self.df[self.df['ImageId']==(name.replace('.png', '').replace('.jpg', ''))]['EncodedPixels']

        if rle.values[0] == '-1':
            masks = np.zeros((1024, 1024))
            cls_label = torch.FloatTensor([0])
        elif rle.values[0] == '2':
            masks = np.zeros((1024, 1024))
            cls_label = torch.FloatTensor([1])            
        else:
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



def generate_dataset_loader_cls_seg(df_all, c_train, train_transform, train_batch_size, c_val, val_transform, val_batch_size, workers):

    train_dataset = Siim_Dataset_cls_seg_train(df_all, c_train, train_transform)
    val_dataset = Siim_Dataset_cls_seg_val(df_all, c_val, val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,        
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batch_size,        
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False)

    return train_loader, val_loader
