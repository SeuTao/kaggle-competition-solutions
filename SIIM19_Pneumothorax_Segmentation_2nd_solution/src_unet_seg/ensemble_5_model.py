# -*- coding: UTF-8 -*-
# @author hjh
import numpy as np
from tqdm import tqdm
import pandas as pd
import PIL


def mask2rle(img, width, height):
    rle = []
    lastColor = 0
    currentPixel = 0
    runStart = -1
    runLength = 0

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel
                    runLength = 1
                else:
                    rle.append(str(runStart))
                    rle.append(str(runLength))
                    runStart = -1
                    runLength = 0
                    currentPixel = 0
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor
            currentPixel += 1

    return " ".join(rle)


sample_submission = pd.read_csv('../data/competition_data/sample_submission.csv')
result_save_path = '../result/'


pred0 = np.load('./chexpert_ef3_pred.npy')
pred1 = np.load('./ef5_pred.npy')
pred2 = np.load('./se101_pred.npy')
pred3 = np.load('./chexpert_deeplab_pred.npy')
pred4 = np.load('./chexpert_se50_pred.npy')

print(pred0.shape)
print(pred1.shape)
print(pred2.shape)
print(pred3.shape)
print(pred4.shape)
pred = (pred0 + pred1 + pred2 + pred3 + pred4) / 5.
c_test = sample_submission['ImageId'].tolist()

best_thr = 0.4
preds = (pred[:, 0, ...] > best_thr)

# Generate rle encodings (images are first converted to the original size)
rles = []
for p in tqdm(preds):
    im = PIL.Image.fromarray((p.T * 255).astype(np.uint8)).resize((1024, 1024))
    im = np.asarray(im)
    rles.append(mask2rle(im, 1024, 1024))

ids = c_test
sub_df = pd.DataFrame({'ImageId': ids, 'EncodedPixels': rles})
sub_df.loc[sub_df.EncodedPixels == '', 'EncodedPixels'] = '-1'
print(sub_df.head())

sub_df.to_csv(result_save_path + 'ensemble5_5fold_{}.csv'.format(best_thr), index=False)