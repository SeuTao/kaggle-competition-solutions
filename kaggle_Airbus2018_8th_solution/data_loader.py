import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from data_process.augmentation import *
from data_process.aug import *
import pandas as pd

def read_txt(txt):
    f = open(txt, 'r')
    lines = f.readlines()
    f.close()
    return [tmp.strip() for tmp in lines]

class SaltDataset5Fold(Dataset):
    def __init__(self, transform, mode, image_size, fold_index, is_non_empty = False):
        self.transform = transform
        self.mode = mode
        self.image_size = image_size

        self.is_non_empty = is_non_empty
        if self.is_non_empty:
            print('Only with ships!!')

        self.train_image_path = r'/data/shentao/Airbus/AirbusShipDetectionChallenge/train_v2'
        self.train_mask_path =  r'/data/shentao/Airbus/AirbusShipDetectionChallenge/train_mask_png_v2'
        self.test_image_path =  r'/data/shentao/Airbus/AirbusShipDetectionChallenge/test_v2'
        self.fold_index = None
        self.set_mode(mode, fold_index)

    def set_mode(self, mode, fold_index):
        self.mode = mode

        self.fold_index = fold_index
        print('fold index set: ', fold_index)

        if self.mode == 'train':
            train_id = pd.read_csv('/data/shentao/Airbus/code/image_list/train.csv')
            train_id = train_id['id']
            train_id = [tmp + '.jpg' for tmp in train_id]


            if self.is_non_empty:
                self.train_image_with_mask = [tmp for tmp in train_id if os.path.exists(os.path.join(self.train_mask_path, tmp + '.png'))]
                print(len(self.train_image_with_mask))
                self.num_data = len(self.train_image_with_mask)

            else:
                self.train_image_no_mask = [tmp for tmp in train_id if not os.path.exists(os.path.join(self.train_mask_path, tmp+'.png'))]
                self.train_image_with_mask = [tmp for tmp in train_id if os.path.exists(os.path.join(self.train_mask_path, tmp+'.png'))]
                print(len(self.train_image_no_mask))
                print(len(self.train_image_with_mask))

                self.num_data = len(self.train_image_no_mask)+len(self.train_image_with_mask)

        elif self.mode == 'val':
            data = pd.read_csv('/data/shentao/Airbus/code/image_list/valid2_1031.csv')
            self.val_list = data['id']
            self.val_list = [tmp+'.jpg' for tmp in self.val_list]

            if self.is_non_empty:
                self.val_list = [tmp for tmp in self.val_list if os.path.exists(os.path.join(self.train_mask_path, tmp+'.png'))]

            self.num_data = len(self.val_list)
            print(self.num_data)

        elif self.mode == 'test':
            self.test_list = read_txt('./image_list/test.txt')
            self.num_data = len(self.test_list)
            print('set dataset mode: test')

    def __getitem__(self, index):
        if self.fold_index is None:
            print('WRONG!!!!!!! fold index is NONE!!!!!!!!!!!!!!!!!')
            return

        if self.mode == 'train':

            if self.is_non_empty:
                img_tmp = self.train_image_with_mask[index]
                image = cv2.imread(os.path.join(self.train_image_path, img_tmp), 1)
                label = cv2.imread(os.path.join(self.train_mask_path, img_tmp + '.png'), 1)

            else:
                if random.randint(0,1) == 0:
                    random_pos = random.randint(0, len(self.train_image_no_mask)-1)
                    img_tmp = self.train_image_no_mask[random_pos]

                    image = cv2.imread(os.path.join(self.train_image_path, img_tmp),1)
                    label = cv2.imread(os.path.join(self.train_mask_path, img_tmp + '.png'),1)
                else:
                    random_pos = random.randint(0, len(self.train_image_with_mask) - 1)
                    img_tmp = self.train_image_with_mask[random_pos]

                    image = cv2.imread(os.path.join(self.train_image_path, img_tmp), 1)
                    label = cv2.imread(os.path.join(self.train_mask_path, img_tmp + '.png'), 1)

        if self.mode == 'val':
            image = cv2.imread(os.path.join(self.train_image_path, self.val_list[index]),1)
            label = cv2.imread(os.path.join(self.train_mask_path, self.val_list[index]+'.png'),1)

            # if os.path.exists(os.path.join(self.train_mask_path, self.val_list[index]+'.png')):
            #     print(os.path.join(self.train_mask_path, self.val_list[index]+'.png'))

        if self.mode == 'test':
            image_id = self.test_list[index].replace('.png','')
            return image_id


        is_empty = False

        if label is None:
            label = np.zeros([768, 768, 3]).astype(np.uint8)
            is_empty = True

        if self.mode == 'train':
            if random.randint(0, 1) == 0:
                image = cv2.flip(image, 1)
                label = cv2.flip(label, 1)

            if random.randint(0, 1) == 0:
                image = cv2.flip(image, 0)
                label = cv2.flip(label, 0)

            if random.randint(0, 1) == 0:
                image = cv2.transpose(image)
                label = cv2.transpose(label)

            if random.randint(0, 1) == 0:
                image = randomHueSaturationValue(image,
                                               hue_shift_limit=(-30, 30),
                                               sat_shift_limit=(-5, 5),
                                               val_shift_limit=(-15, 15))

                image, label = randomShiftScaleRotate(image, label,
                                                   shift_limit=(-0.1, 0.1),
                                                   scale_limit=(-0.1, 0.1),
                                                   aspect_limit=(-0.1, 0.1),
                                                   rotate_limit=(-0, 0))


        # print(label.shape)
        image = cv2.resize(image,(self.image_size, self.image_size))
        # label = cv2.resize(label, (768, 768))

        image = image.reshape([self.image_size, self.image_size, 3])
        label = label.reshape([768, 768, 3])
        label = label[:,:,0]

        image = np.transpose(image, (2,0,1))
        image = image.reshape([3, self.image_size, self.image_size])
        label = label.reshape([1, 768, 768])

        image = (np.asarray(image).astype(np.float32) - 127.5) / 127.5
        label = np.asarray(label).astype(np.float32) / 255.0
        label[label >= 0.5] = 1
        label[label < 0.5] = 0

        if self.mode == 'val':
            image_path = self.val_list[index]
            return image_path, torch.FloatTensor(image), torch.FloatTensor(label), is_empty

        label = label.reshape([768, 768])
        label = cv2.resize(label,(self.image_size, self.image_size)).reshape([1, self.image_size, self.image_size])

        return torch.FloatTensor(image), torch.FloatTensor(label), is_empty

    def __len__(self):
        return self.num_data


def get_loader(image_size, batch_size, mode='train'):
    """Build and return data loader."""
    dataset = SaltDataset5Fold(None, mode, image_size)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader

def get_5foldloader(image_size, batch_size, fold_index, mode='train', is_non_empty = False):
    """Build and return data loader."""
    dataset = SaltDataset5Fold(None, mode, image_size, fold_index, is_non_empty)

    shuffle = False
    if mode == 'train':
        shuffle = True

    if mode == 'test':
        shuffle = True
        print('test shuffle!!!!!!!!!')

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=32, shuffle=shuffle)
    return data_loader

def split_train_val():
    train_image_path = r'/data2/shentao/DATA/Kaggle/Salt/Kaggle_salt/train/images'
    train_mask_path = r'/data2/shentao/DATA/Kaggle/Salt/Kaggle_salt/train/masks'

    image_list = os.listdir(train_image_path)
    random.shuffle(image_list)
    train_num = int(0.8*len(image_list))

    train_image_list = image_list[0: train_num]
    val_image_list = image_list[train_num:]


    print(len(train_image_list))
    print(len(val_image_list))

    f = open('train.txt','w')
    for image in train_image_list:
        f.write(image + '\n')
    f.close()

    f = open('val.txt','w')
    for image in val_image_list:
        f.write(image + '\n')
    f.close()

    print(len(image_list))
    return

def split_train_val_5fold():
    train_image_path = r'/data/shentao/Airbus/AirbusShipDetectionChallenge_384/train_mask_png'
    image_list = os.listdir(train_image_path)
    random.shuffle(image_list)

    image_list = [tmp.replace('.png','') for tmp in image_list]

    part = int(len(image_list) / 5)

    print(part)
    part_list = []
    for i in range(5):
        image_list_part = image_list[i*part:(i+1)*part]
        part_list.append(image_list_part)

    for i in range(5):
        val_list = part_list[i]
        train_list = []
        for j in range(5):
            if j!=i:
                train_list += part_list[j]

        f = open('./image_list/train_fold'+str(i)+'.txt', 'w')
        for image in train_list:
            f.write(image + '\n')
        f.close()

        f = open('./image_list/val_fold'+str(i)+'.txt', 'w')
        for image in val_list:
            f.write(image + '\n')
        f.close()

        print(len(train_list))
        print(len(val_list))
    print('done')

def split_train_val_5fold_cls():
    train_image_path = r'/data/shentao/Airbus/AirbusShipDetectionChallenge_384/train'
    image_list = os.listdir(train_image_path)
    random.shuffle(image_list)

    part = int(len(image_list) / 5)
    print(part)

    part_list = []
    for i in range(5):
        image_list_part = image_list[i*part:(i+1)*part]
        part_list.append(image_list_part)

    for i in range(5):
        val_list = part_list[i]
        train_list = []
        for j in range(5):
            if j!=i:
                train_list += part_list[j]

        f = open('./image_list/train_cls_fold'+str(i)+'.txt', 'w')
        for image in train_list:
            f.write(image + '\n')
        f.close()

        f = open('./image_list/val_cls_fold'+str(i)+'.txt', 'w')
        for image in val_list:
            f.write(image + '\n')
        f.close()

        print(len(train_list))
        print(len(val_list))
    print('done')

def create_test_list():
    test_image_path = r'/data/shentao/Airbus/AirbusShipDetectionChallenge_384/test'

    image_list = os.listdir(test_image_path)
    random.shuffle(image_list)

    f = open('./image_list/test.txt','w')
    for image in image_list:
        f.write(image + '\n')
    f.close()

    print(len(image_list))

