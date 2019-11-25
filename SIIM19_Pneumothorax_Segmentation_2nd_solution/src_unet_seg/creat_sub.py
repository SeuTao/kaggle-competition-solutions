
import pandas as pd
import os

csv_path1 = './result/submission_0.8877.csv'
csv_path2 = '/home/vip415/HJH/mmdetectionV2/kaggle_siim/result/ensemble5_5fold_0.4.csv'
csv_res_path = './result/'
if not os.path.exists(csv_res_path):
    os.makedirs(csv_res_path)

df1 = pd.read_csv(csv_path1)
df2 = pd.read_csv(csv_path2)


binary_list = df1['ImageId'].tolist()
binary_list2 = df2['ImageId'].tolist()

rle_file = []
count = 0
for img_name in binary_list:
    if df1[df1['ImageId'] == img_name]['EncodedPixels'].values[0] != '-1':
        rle_file.append(df2[df2['ImageId'] == img_name]['EncodedPixels'].values[0])
    else:
        rle_file.append('-1')

csv_file_ = csv_res_path + 'sub_ensemble_5_model_swa_0.4.csv'
df = pd.DataFrame({'ImageId': binary_list, 'EncodedPixels': rle_file})
df = df[['ImageId', 'EncodedPixels']]  # change the column index
df.to_csv(csv_file_, index=False, sep=str(','))