import torch
import tqdm
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
SIGFIGS = 6
import pandas as pd
from settings import *
from utils import *
import random
# ### Hungarian
import scipy
import scipy.special
import scipy.optimize

#############################################################
plate = pd.read_csv('./leaks/plate_groups.csv')
assign = pd.read_csv('./leaks/exp_to_group.csv')
plate_dict = {}
plate_dict[0] = plate['p0']
plate_dict[1] = plate['p1']
plate_dict[2] = plate['p2']
plate_dict[3] = plate['p3']
exp_to_group = assign.assignment.values
all_test_exp = assign.experiment.values
a_dict = {}

for a_tmp, test_exp in zip(exp_to_group, all_test_exp):
    a_dict[test_exp] = a_tmp
#############################################################

def _get_predicts(predicts, coefficients):
    return torch.einsum("ij,j->ij", (predicts, coefficients))

def _get_labels_distribution(predicts, coefficients):
    predicts = _get_predicts(predicts, coefficients)
    labels = predicts.argmax(dim=-1)
    counter = torch.bincount(labels, minlength=predicts.shape[1])
    return counter

def _compute_score_with_coefficients(predicts, coefficients):
    counter = _get_labels_distribution(predicts, coefficients).float()
    counter = counter * 100 / len(predicts)
    max_scores = torch.ones(len(coefficients)).cuda().float() * 100 / len(coefficients)
    result, _ = torch.min(torch.cat([counter.unsqueeze(0), max_scores.unsqueeze(0)], dim=0), dim=0)

    return float(result.sum().cpu())

def _find_best_coefficients(predicts, coefficients, alpha=0.001, iterations=100):
    best_coefficients = coefficients.clone()
    best_score = _compute_score_with_coefficients(predicts, coefficients)

    for _ in range(iterations):
        counter = _get_labels_distribution(predicts, coefficients)
        label = int(torch.argmax(counter).cpu())
        coefficients[label] -= alpha
        score = _compute_score_with_coefficients(predicts, coefficients)

        if score > best_score:
            best_score = score
            best_coefficients = coefficients.clone()

    return best_coefficients

def pre_balance(pre, iter = 3000, is_show = True):
    ############ input: pre like this [n_sample,n_class]  cuda tensor
    ##########output  the same as pre ,but it is numpt tensor
    y = pre
    start_alpha=0.01
    alpha = start_alpha
    min_alpha=0.0001

    coefs = torch.ones(y.shape[1]).cuda().float()
    last_score = _compute_score_with_coefficients(y, coefs)

    if is_show:
        print("Start score", last_score)

    while alpha >= min_alpha:
        coefs = _find_best_coefficients(y, coefs, iterations=iter, alpha=alpha)
        new_score = _compute_score_with_coefficients(y, coefs)

        if new_score <= last_score:
            alpha *= 0.5

        last_score = new_score

        if last_score == 100:
            break

        if is_show:
            print("Score: {}, alpha: {}".format(last_score, alpha))

    predicts = _get_predicts(y, coefs)
    return predicts.cpu().numpy()

def from_numpy(predict):

    y = torch.FloatTensor(predict).cuda()
    start_alpha=0.01
    alpha = start_alpha
    min_alpha=0.0001

    coefs = torch.ones(y.shape[1]).cuda().float()
    last_score = _compute_score_with_coefficients(y, coefs)
    print("Start score", last_score)

    while alpha >= min_alpha:
        coefs = _find_best_coefficients(y, coefs, iterations=3000, alpha=alpha)
        new_score = _compute_score_with_coefficients(y, coefs)

        if new_score <= last_score:
            alpha *= 0.5

        last_score = new_score
        print("Score: {}, alpha: {}".format(last_score, alpha))

    predicts = _get_predicts(y, coefs)
    return predicts.cpu().numpy()

def pre_balance_batch_probability(preds1,ids):
    ########### input : preds1 is like thie [n_sample,n_class],ids is predictived ids
    ########### output :pre is  prediction of class,ids is predictived id
    pixel_mean={'id_code':ids}
    submission = pd.DataFrame(pixel_mean,columns=['id_code'])
    submission ['experiment'] = submission.id_code.str.split("_").apply(lambda a: a[0])
    exper=submission['experiment']
    ee=exper.drop_duplicates()
    pre_id=[]
    pre=[]

    for index1 in tqdm.tqdm(ee):

        tmp=submission[submission['experiment']==index1].index.tolist()
        st=tmp[0]
        end=tmp[-1]

        tmp_pre=preds1[st:end+1]
        id_tm=ids[st:end+1]

        tmp_pre_torch= torch.from_numpy(np.array(tmp_pre)).cuda().float()
        tmp=pre_balance(tmp_pre_torch)

        pre_id.extend(id_tm)
        pre.extend(tmp.tolist())

    return pre,pre_id

def pre_balance_plate_probability(preds1, ids, a_dict, a_index_dict, only_public = True, is_softmax = True):
    ########### input : preds1 is like thie [n_sample,n_class],ids is predictived ids
    ########### output :pre is  prediction of class,ids is predictived id

    pixel_mean={'id_code': ids}
    submission = pd.DataFrame(pixel_mean, columns=['id_code'])
    submission ['experiment'] = submission.id_code.str.split("_").apply(lambda a: a[0]+'_'+a[1])
    exper=submission['experiment']
    ee=exper.drop_duplicates()
    pre_id=[]
    pre=[]

    public_set = {'HUVEC-17','HEPG2-08','RPE-08','U2OS-04'}

    for index1 in tqdm.tqdm(ee):
        a_index = a_index_dict[index1.split('_')[0]]
        assignment = a_dict[int(a_index)]
        print('assignment',a_index)

        plate = int(index1.split('_')[1])
        print(index1)
        print(plate)
        plate_label_index = list(assignment == plate)

        tmp=submission[submission['experiment']==index1].index.tolist()
        st=tmp[0]
        end=tmp[-1]
        pre_tmp=np.array(preds1[st:end+1])
        id_tm=ids[st:end+1]
        pre_tmp = np.array(pre_tmp[:,plate_label_index])

        if is_softmax:
            pre_tmp = softmax_np(pre_tmp)

        tmp_pre_torch= torch.from_numpy(pre_tmp).cuda().float()

        if only_public:
            if index1.split('_')[0] in public_set :
                print('IN THE PUBLIC!!!!!!!!!!!')
                tmp=pre_balance(tmp_pre_torch, iter = 3000)
            else:
                tmp=pre_balance(tmp_pre_torch, iter = 3000, is_show = False)
        else:
            tmp = pre_balance(tmp_pre_torch, iter=3000)

        pre_id.extend(id_tm)
        final_predict = tmp.reshape([len(id_tm), 277])
        predict_tmp = np.zeros([len(id_tm), 1108])
        predict_tmp[:,plate_label_index] = final_predict
        pre.extend(predict_tmp.tolist())

    return pre, pre_id

def softmax_np(x):
    re = np.exp(x) / np.sum(np.exp(x), axis=0)
    return re

def balance_plate_probability_training(preds1, ids, a_dict, a_index_dict, iters = 3000, is_show = True):
    ########### input : preds1 is like thie [n_sample,n_class],ids is predictived ids
    ########### output :pre is  prediction of class,ids is predictived id

    pixel_mean={'id_code': ids}
    submission = pd.DataFrame(pixel_mean, columns=['id_code'])
    submission ['experiment'] = submission.id_code.str.split("_").apply(lambda a: a[0]+'_'+a[1])
    exper=submission['experiment']
    ee=exper.drop_duplicates()
    pre_id=[]
    pre=[]

    for index1 in tqdm.tqdm(ee):
        # print(index1)
        a_index = a_index_dict[index1.split('_')[0]]
        assignment = a_dict[int(a_index)]
        # print('assignment',a_index)
        plate = int(index1.split('_')[1])
        # print(plate)
        plate_label_index = list(assignment == plate)

        tmp=submission[submission['experiment']==index1].index.tolist()
        st=tmp[0]
        end=tmp[-1]
        pre_tmp=np.array(preds1[st:end+1])
        id_tm=ids[st:end+1]
        pre_tmp = np.array(pre_tmp[:,plate_label_index])
        pre_id.extend(id_tm)

        pre_tmp = softmax_np(pre_tmp)

        if iters > 0:
            tmp_pre_torch= torch.from_numpy(pre_tmp).cuda().float()
            tmp=pre_balance(tmp_pre_torch, iter = iters,  is_show = is_show)
        else:
            tmp = pre_tmp

        final_predict = tmp.reshape([len(id_tm), 277])
        predict_tmp = np.zeros([len(id_tm), 1108])
        predict_tmp[:,plate_label_index] = final_predict
        pre.extend(predict_tmp.tolist())

    return pre, pre_id

def split_277_acc(preds1, ids, val_label, a_dict, a_index_dict, is_softmax = True, remove_list = {}):
    ########### input : preds1 is like thie [n_sample,n_class],ids is predictived ids
    ########### output :pre is  prediction of class,ids is predictived id

    pixel_mean={'id_code': ids}
    submission = pd.DataFrame(pixel_mean, columns=['id_code'])
    submission ['experiment'] = submission.id_code.str.split("_").apply(lambda a: a[0]+'_'+a[1])
    exper=submission['experiment']
    ee=exper.drop_duplicates()
    pre_id=[]
    pre=[]
    label = []

    # public_set = {'HUVEC-17','HEPG2-08','RPE-08','U2OS-04'}

    for index1 in ee:
        a_index = a_index_dict[index1.split('_')[0]]
        assignment = a_dict[int(a_index)]

        # print(index1.split('_')[0])
        plate = int(index1.split('_')[1])
        plate_label_index = list(assignment == plate)

        tmp=submission[submission['experiment']==index1].index.tolist()
        st=tmp[0]
        end=tmp[-1]
        pre_tmp=np.array(preds1[st:end+1])
        id_tm=ids[st:end+1]
        lb_tm=val_label[st:end+1]

        pre_tmp = np.array(pre_tmp[:,plate_label_index])

        if is_softmax:
            pre_tmp = softmax_np(pre_tmp)

        tmp = np.asarray(pre_tmp)
        final_predict = tmp.reshape([len(id_tm), 277])
        predict_tmp = np.zeros([len(id_tm), 1108])

        if index1.split('_')[0] not in remove_list:
            predict_tmp[:, plate_label_index] = final_predict
            # continue

        pre_id.extend(id_tm)
        pre.extend(predict_tmp.tolist())
        label.extend(lb_tm)

    return np.asarray(pre), pre_id, label

def assign_plate(plate):
    probabilities = np.array(plate)
    cost = probabilities * -1
    rows, cols = scipy.optimize.linear_sum_assignment(cost)
    chosen_elements = set(zip(rows.tolist(), cols.tolist()))

    for sample in range(cost.shape[0]):
        for sirna in range(cost.shape[1]):
            if (sample, sirna) not in chosen_elements:
                probabilities[sample, sirna] = 0

    return probabilities.argmax(axis=1).tolist()

def final_pre_balance_plate_probability(preds1, ids, a_dict, a_index_dict, is_softmax = True):
    ########### input : preds1 is like thie [n_sample,n_class],ids is predictived ids
    ########### output :pre is  prediction of class,ids is predictived id
    pixel_mean={'id_code': ids}
    submission = pd.DataFrame(pixel_mean, columns=['id_code'])
    submission ['experiment'] = submission.id_code.str.split("_").apply(lambda a: a[0]+'_'+a[1])
    exper=submission['experiment']
    ee=exper.drop_duplicates()
    pre_id=[]
    pre=[]

    for index1 in ee:
        a_index = a_index_dict[index1.split('_')[0]]
        assignment = a_dict[int(a_index)]
        print('assignment',a_index)

        plate = int(index1.split('_')[1])
        print(index1)
        print(plate)
        plate_label_index = list(assignment == plate)

        tmp=submission[submission['experiment']==index1].index.tolist()
        st=tmp[0]
        end=tmp[-1]
        pre_tmp=np.array(preds1[st:end+1])
        id_tm=ids[st:end+1]
        pre_tmp = np.array(pre_tmp[:,plate_label_index])

        if is_softmax:
            pre_tmp = softmax_np(pre_tmp)

        tmp_pre_torch= torch.from_numpy(pre_tmp).cuda().float()
        tmp = pre_balance(tmp_pre_torch, iter=3000, is_show = True)

        pre_id.extend(id_tm)
        final_predict = tmp.reshape([len(id_tm), 277])
        predict_tmp = np.zeros([len(id_tm), 1108])
        predict_tmp[:,plate_label_index] = final_predict

        predict_tmp = assign_plate(predict_tmp)
        pre.extend(predict_tmp)

    return pre, pre_id

def ave_balance():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    name = r'se__finetune_cell_fold0-4'
    avg = read_models(model_pred)
    csv_name = write_models(avg, name, is_top1=True)

    n_samples = len(avg)
    predict = np.zeros([n_samples, 1108])
    ids = []
    i = 0
    for id in avg:
        sum = 0.0
        for index in avg[id]:
            predict[i, index] = avg[id][index]
            sum += avg[id][index]

        predict[i] = predict[i] / sum
        ids.append(id)
        i += 1

    predict = predict.tolist()
    prob_to_csv_top5(predict, ids, name+'_top5.csv')
    predict, ids = pre_balance_batch_probability(predict, ids)
    prob_to_csv_top5(predict, ids, name+'_batch_balance_top5.csv')

    prob = np.asarray(predict)
    print(prob.shape)
    top = np.argsort(-prob,1)[:,:5]
    index = 0
    rs = []

    for (t0,t1,t2,t3,t4) in top:
        top_k_label_name = r''
        top_k_label_name += str(t0)
        index += 1
        rs.append(top_k_label_name)

    pd.DataFrame({'id_code':ids, 'sirna':rs}).to_csv( '{}.csv'.format(name+'_batch_balance'), index=None)

def balance_ave():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    for item in model_pred:
        weight = model_pred[item]
        tmp_dict = {}
        tmp_dict[item] = weight
        avg = read_models(tmp_dict)

        n_samples = len(avg)
        predict = np.zeros([n_samples, 1108])
        ids = []
        i = 0

        for id in avg:
            sum = 0.0
            for index in avg[id]:
                predict[i, index] = avg[id][index]
                sum += avg[id][index]

            predict[i] = predict[i] / sum
            ids.append(id)
            i += 1

        predict = predict.tolist()
        predict, ids = pre_balance_batch_probability(predict, ids)
        prob_to_csv_top5(predict, ids, './ave_predict/'+os.path.split(item)[1])

    # prob = np.asarray(predict)
    # print(prob.shape)
    # top = np.argsort(-prob,1)[:,:5]
    # index = 0
    # rs = []
    #
    # for (t0,t1,t2,t3,t4) in top:
    #     top_k_label_name = r''
    #     top_k_label_name += str(t0)
    #     index += 1
    #     rs.append(top_k_label_name)
    #
    # pd.DataFrame({'id_code':ids, 'sirna':rs}).to_csv( '{}.csv'.format(name+'_batch_balance'), index=None)

def ave_plate_balance():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    name = r'tmp'
    avg = read_models(model_pred)

    n_samples = len(avg)
    predict = np.zeros([n_samples, 1108])
    ids = []
    i = 0
    for id in avg:
        sum = 0.0
        for index in avg[id]:
            predict[i, index] = avg[id][index]
            sum += avg[id][index]

        predict[i] = predict[i] / sum
        ids.append(id)
        i += 1

    predict = predict.tolist()
    predict, ids = pre_balance_plate_probability(predict, ids)

    prob = np.asarray(predict)
    print(prob.shape)
    top = np.argsort(-prob,1)[:,:5]
    index = 0
    rs = []

    for (t0,t1,t2,t3,t4) in top:
        top_k_label_name = r''
        top_k_label_name += str(t0)
        index += 1
        rs.append(top_k_label_name)

    pd.DataFrame({'id_code':ids, 'sirna':rs}).to_csv( '{}.csv'.format(name+'_batch_balance'), index=None)

def filter_csv(csv, cell):
    df = pd.read_csv(csv)

    id_code = df['id_code']
    siRNA = df['sirna']

    siRNA_tmp = []

    for id_tmp, rna_tmp in zip(id_code, siRNA):
        if cell not in id_tmp:
            print(id_tmp)
            print(rna_tmp)
            siRNA_tmp.append(1138)
        else:
            siRNA_tmp.append(rna_tmp)

    pd.DataFrame({'id_code': id_code, 'sirna': siRNA_tmp}).to_csv(csv.replace('.csv','_with_all.csv'),index=None)

def replace(cell, origin, new, csv ):
    df = pd.read_csv(origin)
    id_code = df['id_code']
    siRNA = df['sirna']

    id_code_tmp = []
    siRNA_tmp = []

    for id_tmp, rna_tmp in zip(id_code, siRNA):
        if cell in id_tmp:
            continue
        else:
            siRNA_tmp.append(rna_tmp)
            id_code_tmp.append(id_tmp)

    df = pd.read_csv(new)
    id_code = df['id_code']
    siRNA = df['sirna']

    for id_tmp, rna_tmp in zip(id_code, siRNA):
        if cell in id_tmp:
            siRNA_tmp.append(rna_tmp)
            id_code_tmp.append(id_tmp)
            print(id_tmp)
            print(rna_tmp)

    pd.DataFrame({'id_code': id_code_tmp, 'sirna': siRNA_tmp}).to_csv(csv,index=None)
    return csv

def get_valid_label(fold = 0):
    df = pd.read_csv(path_data + '/valid_fold_'+str(fold)+'.csv')
    cell = 'U2OS'
    if cell is not None:
        print('cell type: ' + cell)
        df_all = []
        for i in range(100):
            df_ = pd.DataFrame(df.loc[df['experiment'] == (cell + '-' + str(100 + i)[-2:])])
            df_all.append(df_)
        df = pd.concat(df_all)

    val_label = np.asarray(list(df[r'sirna'])).reshape([-1])

    return val_label

def get_test_id(cell):
    df = pd.read_csv(data_dir + '/test.csv')

    if cell is not None:
        df_all = []
        for i in range(100):
            df_ = pd.DataFrame(df.loc[df['experiment'] == (cell + '-' + str(100 + i)[-2:])])
            df_all.append(df_)
        df = pd.concat(df_all)

    return list(df[r'id_code'])

def get_npy( dir, cell='HUVEC', mode='test.npy'):

    ids = get_test_id(cell=cell)
    print(len(ids))

    all_list = []
    dir_list = os.listdir(dir)

    for dir_tmp in dir_list:
        npy_tmp = os.path.join(dir, dir_tmp)
        if cell in npy_tmp and  mode in npy_tmp:
            all_list.append(npy_tmp)

    tmp_list = []
    for npy in all_list:
        tmp = np.load(npy)
        tmp_list.append(np.asarray(tmp))

    tmp = np.mean(tmp_list, axis=0)
    return tmp

if __name__ == '__main__':
    import os

    only_public = False

    for cell in ['U2OS', 'RPE', 'HEPG2', 'HUVEC']:
        ids = get_test_id(cell=cell)

        test =  get_npy(dir=model_save_path+r'/xception_large_6channel_512_fold0_final/checkpoint/max_valid_model_'
                            +cell+r'_semi_snapshot_all_npy',
                        cell=cell, mode='test.npy')

        predict, ids = pre_balance_plate_probability(test, ids, plate_dict, a_dict, only_public=False)
        prob = np.asarray(predict)

        print(prob.shape)
        top = np.argsort(-prob, 1)[:, :5]
        index = 0
        rs = []

        for (t0, t1, t2, t3, t4) in top:
            top_k_label_name = r''
            top_k_label_name += str(t0)
            rs.append(top_k_label_name)

        name = cell
        pd.DataFrame({'id_code': ids, 'sirna': rs}).to_csv('{}.csv'.format(name), index=None)
        print(name)


    csv = replace(cell = 'RPE',
                  origin = r'./process/sample_submission.csv',
                  new = 'RPE.csv',
                  csv = 'submission.csv')

    csv = replace(cell = 'HEPG2',
                  origin = csv,
                  new = 'HEPG2.csv',
                  csv = csv)

    csv = replace(cell = 'HUVEC',
                  origin = csv,
                  new ='HUVEC.csv',
                  csv = csv)

    csv = replace(cell = 'U2OS',
                  origin = csv,
                  new = 'U2OS.csv',
                  csv = csv)

    os.remove('./U2OS.csv')
    os.remove('./RPE.csv')
    os.remove('./HEPG2.csv')
    os.remove('./HUVEC.csv')
    print(csv)