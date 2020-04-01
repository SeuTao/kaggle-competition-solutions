import numpy as np
import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.utils.data as D
import torch.nn.functional as F
import torchvision
from sklearn.model_selection import KFold
from tqdm import tqdm
import cv2
from torchvision import transforms as T
from process.augmentation import *
from settings import *


id_index_dict = {'HEPG':0,'HUVEC':1,'RPE':2,'U2OS':3,}

class Dataset_(D.Dataset):
    def __init__(self, csv_file, img_dir, mode='train', cell= None, image_size=512, site=1,
                 augment = [],
                 rgb=False,
                 pseudo_csv = None,
                 # Pseudo_Batch=r'U2OS-05',
                 augment_param_dict = None,
                 is_TransTwice=False):

        print(csv_file)

        self.channels = [1, 2, 3, 4, 5, 6]
        df = pd.read_csv(csv_file)
        self.image_size = image_size
        self.is_TransTwice = is_TransTwice

        self.site = site
        self.mode = mode
        self.img_dir = img_dir

        self.augment = augment
        self.rgb = rgb
        self.augment_param_dict = augment_param_dict

        if cell is not None:
            print('cell type: '+ cell)
            df_all = []
            for i in range(100):
                df_ = pd.DataFrame(df.loc[df['experiment'] == (cell+'-'+str(100+i)[-2:])])
                df_all.append(df_)
            df  = pd.concat(df_all)

        self.records = df.to_records(index=False)
        self.len = self.records.shape[0]
        self.mean_std = pd.read_csv(os.path.join(mean_std_path, r'mean_std_plate.csv'))

        print(self.len)

    @staticmethod
    def _load_img_as_tensor(file_name):
        with Image.open(file_name) as img:
            return T.ToTensor()(img)

    def _get_img_path(self, index, channel):

        experiment, well, plate = self.records[index].experiment, self.records[index].well, self.records[index].plate
        path =  '/'.join([self.img_dir, 'train', experiment, f'Plate{plate}', f'{well}_s{self.site}_w{channel}.png'])

        if not os.path.exists(path):
            experiment, well, plate = self.records[index].experiment, self.records[index].well, self.records[
                index].plate
            path =  '/'.join([self.img_dir, 'test', experiment, f'Plate{plate}', f'{well}_s{self.site}_w{channel}.png'])

        return path


    def __getitem__(self, index):

        if self.mode == 'train' or 'semi' in self.mode:
            if np.random.random() < 0.5:
                self.site = 2
            else:
                self.site = 1

        experiment, plate = self.records[index].experiment, self.records[index].plate

        def get_img(paths):
            mean = []
            std = []
            for ch in self.channels:
                tmp = self.mean_std[(self.mean_std['experiment'] == experiment) & (self.mean_std['plate'] == plate) & (
                        self.mean_std['channel'] == ch)]['mean'].values
                tmp1 = self.mean_std[(self.mean_std['experiment'] == experiment) & (self.mean_std['plate'] == plate) & (
                        self.mean_std['channel'] == ch)]['std'].values
                mean.append(tmp)
                std.append(tmp1)

            img = []
            for j, img_path in enumerate(paths):
                tmp_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                tmp_img = cv2.resize(tmp_img, (self.image_size, self.image_size))

                if self.mode == 'train' or 'semi' in self.mode:
                    m_r = 0.1 * random.uniform(-1,1)
                    s_r = 0.1 * random.uniform(-1,1)

                    mean[j] = mean[j]*(1+m_r)
                    std[j] = std[j]*(1+s_r)

                tmp_img = (tmp_img - mean[j]) / std[j]
                img.append(tmp_img)

            img = np.array(img)
            image = np.transpose(img, (1, 2, 0))
            return image

        paths = [self._get_img_path(index, ch) for ch in self.channels]
        image  = get_img(paths)

        if self.is_TransTwice:
            image_hard = image.copy()
            image_hard = aug_image(image_hard, type = 'hard')
            img_hard= np.transpose(image_hard, (2, 0, 1))
            img_hard = img_hard.reshape([6, self.image_size, self.image_size])
            img_hard = torch.FloatTensor(img_hard)

            image_easy = image.copy()
            image_easy = aug_image(image_easy, type = 'easy')
            img_easy = np.transpose(image_easy, (2, 0, 1))
            img_easy=  img_easy.reshape([6, self.image_size, self.image_size])
            img_easy = torch.FloatTensor(img_easy)

            image = aug_image(image, type = 'normal')
        else:
            if self.mode == 'train':
                image = aug_image(image, type = 'normal')
            else:
                image = aug_image(image, is_infer=True, augment=self.augment)

        img = np.transpose(image, (2, 0, 1))
        img = img.reshape([6, self.image_size, self.image_size])
        img = torch.FloatTensor(img)

        if self.mode == 'train'  or self.mode == 'valid' or self.mode == 'semi_valid':
            label = torch.from_numpy(np.array(self.records[index].sirna))
            id_code = self.records[index].id_code

            if self.mode == 'semi_valid':
                label = torch.from_numpy(np.array(-1))

            if self.is_TransTwice:
                return img, img_easy, img_hard, label, id_code

            return img, label, id_code

        elif self.mode == 'semi_test' and self.is_TransTwice:
            label = torch.from_numpy(np.array(-1))
            id_code = self.records[index].id_code
            return img, img_easy, img_hard, label, id_code

        else:
            return self.records[index].id_code, img

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len

def run_check_train_data():
    dataset = Dataset_(path_data + r'/train_fold_0.csv',
                       data_dir,
                       cell='U2OS',
                       image_size=256,
                       rgb=False)

    print(dataset)
    num = len(dataset)
    for m in range(num):
        i = np.random.choice(num)
        image, label, id_index = dataset[i]
        print(image.shape)
        print(label)
        print(id_index)

def run_check_test_data():
    dataset = Dataset_(data_dir+'/test.csv', data_dir, mode='semi_test',is_TransTwice=True)
    print(dataset)

    num = len(dataset)
    for m in range(num):
        i = np.random.choice(num)
        image,image_pwc,id,_ = dataset[i]

        print(image.size())
        print(image_pwc.size())
        print(id)
        # print(id)

if __name__ == '__main__':
    run_check_train_data()
