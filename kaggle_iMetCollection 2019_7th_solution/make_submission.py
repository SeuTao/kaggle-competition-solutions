import argparse
import pandas as pd
from utils import mean_df
from dataset import DATA_ROOT
from main import binarize_prediction
import os

def get_predicts(list, weights):
    all = []
    for tmp,w in zip(list,weights):
        tmp_list = os.listdir(tmp)
        tmp_list =[[os.path.join(tmp,tmp_file),w] for tmp_file in tmp_list]
        all.extend(tmp_list)
    return all

def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--threshold', type=float, default=0.1)
    args = parser.parse_args()

    args.output = './all_tmp.csv'
    sample_submission = pd.read_csv(DATA_ROOT + '/' + 'sample_submission.csv', index_col='id')

    dirs = [

        r'/data1/shentao/competitions_py3/imet/result/se101_384_ratio_0.6_0.99_re_fold0/112tta',
        r'/data1/shentao/competitions_py3/imet/result/se101_384_ratio_0.6_0.99_re_fold1/112tta',
        r'/data1/shentao/competitions_py3/imet/result/se101_384_ratio_0.6_0.99_re_fold2/112tta',
        r'/data1/shentao/competitions_py3/imet/result/se101_384_ratio_0.6_0.99_re_fold3/112tta',
        r'/data1/shentao/competitions_py3/imet/result/se101_384_ratio_0.6_0.99_re_fold4/112tta',
        r'/data1/shentao/competitions_py3/imet/result/se101_384_ratio_0.6_0.99_re_fold5/112tta',
        r'/data1/shentao/competitions_py3/imet/result/se101_384_ratio_0.6_0.99_re_fold6/112tta',
        r'/data1/shentao/competitions_py3/imet/result/se101_384_ratio_0.6_0.99_re_fold7/112tta',
        r'/data1/shentao/competitions_py3/imet/result/se101_384_ratio_0.6_0.99_re_fold8/112tta',
        r'/data1/shentao/competitions_py3/imet/result/se101_384_ratio_0.6_0.99_re_fold9/112tta',

        r'/data1/shentao/competitions_py3/imet/result/se50_384_ratio_0.6_0.99_re_fold0/112tta',
        r'/data1/shentao/competitions_py3/imet/result/se50_384_ratio_0.6_0.99_re_fold1/112tta',
        r'/data1/shentao/competitions_py3/imet/result/se50_384_ratio_0.6_0.99_re_fold2/112tta',
        r'/data1/shentao/competitions_py3/imet/result/se50_384_ratio_0.6_0.99_re_fold3/112tta',
        r'/data1/shentao/competitions_py3/imet/result/se50_384_ratio_0.6_0.99_re_fold4/112tta',
        r'/data1/shentao/competitions_py3/imet/result/se50_384_ratio_0.6_0.99_re_fold5/112tta',
        r'/data1/shentao/competitions_py3/imet/result/se50_384_ratio_0.6_0.99_re_fold6/112tta',
        r'/data1/shentao/competitions_py3/imet/result/se50_384_ratio_0.6_0.99_re_fold7/112tta',
        r'/data1/shentao/competitions_py3/imet/result/se50_384_ratio_0.6_0.99_re_fold8/112tta',
        r'/data1/shentao/competitions_py3/imet/result/se50_384_ratio_0.6_0.99_re_fold9/112tta',

        r'/data1/shentao/competitions_py3/imet/result/venn/112tta_se101_skf_fold0',
        r'/data1/shentao/competitions_py3/imet/result/venn/112tta_se101_skf_fold1',
        r'/data1/shentao/competitions_py3/imet/result/venn/112tta_se101_skf_fold2',
        r'/data1/shentao/competitions_py3/imet/result/venn/112tta_se101_skf_fold3',
        r'/data1/shentao/competitions_py3/imet/result/venn/112tta_se101_skf_fold4',
        r'/data1/shentao/competitions_py3/imet/result/venn/112tta_se101_skf_fold5',
        r'/data1/shentao/competitions_py3/imet/result/venn/112tta_se101_skf_fold6',
        r'/data1/shentao/competitions_py3/imet/result/venn/112tta_se101_skf_fold7',
        r'/data1/shentao/competitions_py3/imet/result/venn/112tta_se101_skf_fold8',
        r'/data1/shentao/competitions_py3/imet/result/venn/112tta_se101_skf_fold9',
    ]

    weight = [
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
    ]

    predicts = get_predicts(dirs,weight)
    print(len(predicts))
    dfs = []

    sum_w = 0
    for prediction,w in predicts:
        print('pred', prediction)
        df = pd.read_hdf(prediction, index_col='id')
        df = df.reindex(sample_submission.index)*w
        sum_w += w
        dfs.append(df)

    ratio = sum_w / len(predicts)
    print(ratio)

    df = pd.concat(dfs)
    df = mean_df(df) / ratio
    df[:] = binarize_prediction(df.values, threshold=args.threshold)
    df = df.apply(get_classes, axis=1)
    df.name = 'attribute_ids'
    df.to_csv(args.output, header=True)


def get_classes(item):
    return ' '.join(cls for cls, is_present in item.items() if is_present)


if __name__ == '__main__':
    main()
