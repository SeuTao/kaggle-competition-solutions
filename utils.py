import logging
import os
# import pathlib
import random
import sys
import time
from itertools import chain
from collections import Iterable

# from deepsense import neptune
import numpy as np
import pandas as pd
import torch
from PIL import Image
# import matplotlib.pyplot as plt
# from attrdict import AttrDict
from tqdm import tqdm
from pycocotools import mask as cocomask
from sklearn.model_selection import BaseCrossValidator
# from steppy.base import BaseTransformer
import yaml
from imgaug import augmenters as iaa
import imgaug as ia

# NEPTUNE_CONFIG_PATH = str(pathlib.Path(__file__).resolve().parents[1] / 'configs' / 'neptune.yaml')

def state_dict_remove_moudle(moudle_state_dict, model):
    state_dict = model.state_dict()
    keys = list(moudle_state_dict.keys())
    for key in keys:
        print(key + ' loaded')
        new_key = key.replace(r'module.', r'')
        print(new_key)
        state_dict[new_key] = moudle_state_dict[key]

    return state_dict

def state_dict_add_moudle(moudle_state_dict, model):
    state_dict = model.state_dict()
    keys = list(moudle_state_dict.keys())
    for key in keys:
        print(key + ' loaded')
        new_key = 'module.'+key
        print(new_key)
        state_dict[new_key] = moudle_state_dict[key]

    return state_dict

# Alex Martelli's 'Borg'
# http://python-3-patterns-idioms-test.readthedocs.io/en/latest/Singleton.html
# class _Borg:
#     _shared_state = {}
#
#     def __init__(self):
#         self.__dict__ = self._shared_state


# class NeptuneContext(_Borg):
#     def __init__(self, fallback_file=NEPTUNE_CONFIG_PATH):
#         _Borg.__init__(self)
#
#         self.ctx = neptune.Context()
#         self.fallback_file = fallback_file
#         self.params = self._read_params()
#         self.numeric_channel = neptune.ChannelType.NUMERIC
#         self.image_channel = neptune.ChannelType.IMAGE
#         self.text_channel = neptune.ChannelType.TEXT
#
#     def channel_send(self, *args, **kwargs):
#         self.ctx.channel_send(*args, **kwargs)
#
#     def _read_params(self):
#         if self.ctx.params.__class__.__name__ == 'OfflineContextParams':
#             params = self._read_yaml().parameters
#         else:
#             params = self.ctx.params
#         return params
#
#     def _read_yaml(self):
#         with open(self.fallback_file) as f:
#             config = yaml.load(f)
#         return AttrDict(config)


def init_logger():
    logger = logging.getLogger('salt-detection')
    logger.setLevel(logging.INFO)
    message_format = logging.Formatter(fmt='%(asctime)s %(name)s >>> %(message)s',
                                       datefmt='%Y-%m-%d %H-%M-%S')

    # console handler for validation info
    ch_va = logging.StreamHandler(sys.stdout)
    ch_va.setLevel(logging.INFO)

    ch_va.setFormatter(fmt=message_format)

    # add the handlers to the logger
    logger.addHandler(ch_va)

    return logger


def get_logger():
    return logging.getLogger('salt-detection')


def decompose(labeled):
    nr_true = labeled.max()
    masks = []
    for i in range(1, nr_true + 1):
        msk = labeled.copy()
        msk[msk != i] = 0.
        msk[msk == i] = 255.
        masks.append(msk)

    if not masks:
        return [labeled]
    else:
        return masks

def encode_rle(predictions):
    return [run_length_encoding(mask) for mask in predictions]


# def read_masks(masks_filepaths):
#     masks = []
#     for mask_filepath in tqdm(masks_filepaths):
#         mask = Image.open(mask_filepath)
#         mask = np.asarray(mask.convert('L').point(lambda x: 0 if x < 128 else 1)).astype(np.uint8)
#         masks.append(mask)
#     return masks
#
#
# def read_images(filepaths):
#     images = []
#     for filepath in filepaths:
#         image = np.array(Image.open(filepath))
#         images.append(image)
#     return images

# def create_submission(meta, predictions):
#     output = []
#     for image_id, mask in zip(meta['id'].values, predictions):
#         rle_encoded = ' '.join(str(rle) for rle in run_length_encoding(mask))
#         output.append([image_id, rle_encoded])
#
#     submission = pd.DataFrame(output, columns=['id', 'rle_mask']).astype(str)
#     return submission
import cv2
def create_submission(predictions, w, h):
    output = []

    for image_id, mask in predictions:
        # print(image_id)
        mask = cv2.resize(mask, (w, h), cv2.INTER_NEAREST)
        rle_encoded = ' '.join(str(rle) for rle in run_length_encoding(mask))
        output.append([image_id, rle_encoded])

    submission = pd.DataFrame(output, columns=['ImageId', 'EncodedPixels']).astype(str)
    return submission

def run_length_encoding(x):
    # https://www.kaggle.com/c/data-science-bowl-2018/discussion/48561#
    if np.max(x) == 0:
        return []

    bs = np.where(x.T.flatten())[0]

    rle = []
    prev = -2
    for b in bs:
        if (b > prev + 1): rle.extend((b + 1, 0))
        rle[-1] += 1
        prev = b

    if len(rle) != 0 and rle[-1] + rle[-2] > (x.size-1):
        rle[-2] = rle[-2] - 1

    return rle


def run_length_decoding(mask_rle, shape):
    """
    Based on https://www.kaggle.com/msl23518/visualize-the-stage1-test-solution and modified
    Args:
        mask_rle: run-length as string formatted (start length)
        shape: (height, width) of array to return

    Returns:
        numpy array, 1 - mask, 0 - background

    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[1] * shape[0], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((shape[1], shape[0])).T

def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def softmax(X, theta=1.0, axis=None):
    """
    https://nolanbconaway.github.io/blog/2017/softmax-numpy
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


def from_pil(*images):
    images = [np.array(image) for image in images]
    if len(images) == 1:
        return images[0]
    else:
        return images


def to_pil(*images):
    images = [Image.fromarray((image).astype(np.uint8)) for image in images]
    if len(images) == 1:
        return images[0]
    else:
        return images

def binary_from_rle(rle):
    return cocomask.decode(rle)


def get_crop_pad_sequence(vertical, horizontal):
    top = int(vertical / 2)
    bottom = vertical - top
    right = int(horizontal / 2)
    left = horizontal - right
    return (top, right, bottom, left)


def get_list_of_image_predictions(batch_predictions):
    image_predictions = []
    for batch_pred in batch_predictions:
        image_predictions.extend(list(batch_pred))
    return image_predictions


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ImgAug:
    def __init__(self, augmenters):
        if not isinstance(augmenters, list):
            augmenters = [augmenters]
        self.augmenters = augmenters
        self.seq_det = None

    def _pre_call_hook(self):
        seq = iaa.Sequential(self.augmenters)
        seq = reseed(seq, deterministic=True)
        self.seq_det = seq

    def transform(self, *images):
        images = [self.seq_det.augment_image(image) for image in images]
        if len(images) == 1:
            return images[0]
        else:
            return images

    def __call__(self, *args):
        self._pre_call_hook()
        return self.transform(*args)


def get_seed():
    seed = int(time.time()) + int(os.getpid())
    return seed


def reseed(augmenter, deterministic=True):
    augmenter.random_state = ia.new_random_state(get_seed())
    if deterministic:
        augmenter.deterministic = True

    for lists in augmenter.get_children_lists():
        for aug in lists:
            aug = reseed(aug, deterministic=True)
    return augmenter