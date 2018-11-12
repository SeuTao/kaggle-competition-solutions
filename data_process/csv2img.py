import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import warnings
warnings.filterwarnings("ignore")

INPUT_PATH = '/data/shentao/Airbus/AirbusShipDetectionChallenge'
# DATA_PATH = INPUT_PATH
TRAIN_DATA = os.path.join(INPUT_PATH, "train_v2")
TRAIN_MASKS_DATA = os.path.join(INPUT_PATH, "train_mask_png_v2")
TEST_DATA = os.path.join(INPUT_PATH, "test_v2")

df = pd.read_csv(INPUT_PATH+'/train_ship_segmentations_v2.csv')
path_train = '/data/shentao/Airbus/AirbusShipDetectionChallenge/train_v2/'
path_test = '/data/shentao/Airbus/AirbusShipDetectionChallenge/test_v2/'
train_ids = df.ImageId.values
df = df.set_index('ImageId')

def get_filename(image_id, image_type):
    check_dir = False
    if "Train" == image_type:
        data_path = TRAIN_DATA
    elif "mask" in image_type:
        data_path = TRAIN_MASKS_DATA
    elif "Test" in image_type:
        data_path = TEST_DATA
    else:
        raise Exception("Image type '%s' is not recognized" % image_type)

    if check_dir and not os.path.exists(data_path):
        os.makedirs(data_path)

    return os.path.join(data_path, "{}".format(image_id))

def get_image_data(image_id, image_type, **kwargs):
    img = _get_image_data_opencv(image_id, image_type, **kwargs)
    img = img.astype('uint8')
    return img

def _get_image_data_opencv(image_id, image_type, **kwargs):
    fname = get_filename(image_id, image_type)
    img = cv2.imread(fname)
    assert img is not None, "Failed to read image : %s, %s" % (image_id, image_type)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

# https://github.com/ternaus/TernausNet/blob/master/Example.ipynb
def mask_overlay(image, mask):
    """
    Helper function to visualize mask
    """
    mask = mask.astype(np.uint8)
    weighted_sum = cv2.addWeighted(mask, 0.75, image, 0.5, 0.)
    img = image.copy()
    ind = mask[:, :, 1] > 0
    img[ind] = weighted_sum[ind]
    return img


# path = r'/data/shentao/Airbus/AirbusShipDetectionChallenge_384/train_mask'
# list = os.listdir(path)
# print(len(list))

nImg = 32  # no. of images that you want to display
np.random.seed(42)
if df.index.name == 'ImageId':
    df = df.reset_index()
if df.index.name != 'ImageId':
    df = df.set_index('ImageId')

_train_ids = list(train_ids)
# np.random.shuffle(_train_ids)

count = 0
for image_id in _train_ids:
    all_masks = np.zeros((768, 768))

    if str(df.loc[image_id, 'EncodedPixels']) == str(np.nan):
        continue

    try:

        img_masks = df.loc[image_id, 'EncodedPixels'].tolist()

        print(len(img_masks))
        for mask in img_masks:
            all_masks += rle_decode(mask)

        all_masks = np.expand_dims(all_masks, axis=2)
        all_masks = np.repeat(all_masks, 3, axis=2).astype('uint8') * 255

        if len(img_masks ) > 10:
            cv2.imwrite('tmp.png', all_masks)
            break

    except Exception as e:

        print('exception')
        # all_masks = rle_decode(df.loc[image_id, 'EncodedPixels'])
        # all_masks = np.expand_dims(all_masks, axis=2) * 255
        # all_masks = np.repeat(all_masks, 3, axis=2).astype('uint8')
    # cv2.imwrite('/data/shentao/Airbus/AirbusShipDetectionChallenge/train_mask_png_v2/'+image_id+'.png',all_masks)

    count += 1
    if count % 100 == 0:
        print(count)






