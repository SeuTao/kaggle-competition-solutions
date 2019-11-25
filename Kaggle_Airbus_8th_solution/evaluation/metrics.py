import numpy as np
from tqdm import tqdm
from pycocotools import mask as cocomask
# from utils import get_segmentations

def iou(gt, pred):
    gt[gt > 0.5] = 1.
    pred[pred > 0.5] = 1.
    gt[gt <= 0.5] = 0.
    pred[pred <= 0.5] = 0.

    intersection = gt * pred
    union = gt + pred
    union[union > 0] = 1.

    intersection = np.sum(intersection)
    union = np.sum(union)

    if union == 0:
        union = 1e-09
    return intersection / float(union)

def binarize(image, threshold):
    image_binarized = (image[1, :, :] > threshold).astype(np.uint8)
    return image_binarized

def rle_from_binary(prediction):
    prediction = np.asfortranarray(prediction)
    return cocomask.encode(prediction)

def get_segmentations(labeled):
    nr_true = labeled.max()
    segmentations = []
    for i in range(1, nr_true + 1):
        msk = labeled == i
        segmentation = rle_from_binary(msk.astype('uint8'))
        segmentation['counts'] = segmentation['counts'].decode("UTF-8")
        segmentations.append(segmentation)
    return segmentations

def compute_ious(gt, predictions):
    gt_ = get_segmentations(gt)
    predictions_ = get_segmentations(predictions)

    if len(gt_) == 0 and len(predictions_) == 0:
        return np.ones((1, 1))
    elif len(gt_) != 0 and len(predictions_) == 0:
        return np.zeros((1, 1))
    else:
        iscrowd = [0 for _ in predictions_]
        ious = cocomask.iou(gt_, predictions_, iscrowd)
        if not np.array(ious).size:
            ious = np.zeros((1, 1))
        return ious

def compute_precision_at(ious, threshold):
    mx1 = np.max(ious, axis=0)
    mx2 = np.max(ious, axis=1)
    tp = np.sum(mx2 >= threshold)
    fp = np.sum(mx2 < threshold)
    fn = np.sum(mx1 < threshold)
    return float(tp) / (tp + fp + fn)


def compute_eval_metric(gt, predictions):
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    ious = compute_ious(gt, predictions)
    precisions = [compute_precision_at(ious, th) for th in thresholds]
    return sum(precisions) / len(precisions)


def intersection_over_union(y_true, y_pred):
    ious = []
    for y_t, y_p in tqdm(list(zip(y_true, y_pred))):
        iou = compute_ious(y_t, y_p)
        iou_mean = 1.0 * np.sum(iou) / len(iou)
        ious.append(iou_mean)
    return np.mean(ious)


def intersection_over_union_thresholds(y_true, y_pred):
    iouts = []
    for y_t, y_p in tqdm(list(zip(y_true, y_pred))):
        iouts.append(compute_eval_metric(y_t, y_p))
    return np.mean(iouts)


# if __name__ == '__main__':
#     import cv2
