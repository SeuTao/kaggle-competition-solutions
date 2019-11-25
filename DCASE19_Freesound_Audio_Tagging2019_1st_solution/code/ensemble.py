from utils import *
from sklearn.metrics import label_ranking_average_precision_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from models import stacker
from keras import backend as K




def stacking(cfg,files):

    print(list(files.keys()))
    ave_oof, ave_pred = average(cfg,files,True)
    tr_oof_files = [np.load(f'../output/{name}oof.npy')[:,:,np.newaxis] for name in files.keys()] + [ave_oof[:,:,np.newaxis]]
    tr_oof = np.concatenate(tr_oof_files,axis=-1)
    test_files = [np.load(f'../output/{name}pred.npy')[:,:,np.newaxis] for name in files.keys()] + [ave_pred[:,:,np.newaxis]]
    test_pred = np.concatenate(test_files,axis=-1)
    df = pd.read_csv(f'../input/train_curated.csv')
    y = split_and_label(df['labels'].values)


    mskfold = MultilabelStratifiedKFold(cfg.n_folds, shuffle=False, random_state=66666)
    folds = list(mskfold.split(y, y))

    predictions = np.zeros_like(test_pred)[:,:,0]
    oof = np.zeros_like((y))
    for fold, (tr_idx, val_idx) in enumerate(folds):
        print('fold ',fold)
        if True:  # init
            K.clear_session()
            model = stacker(cfg,tr_oof.shape[2])
            best_epoch = 0
            best_score = -1

        for epoch in range(1000):
            if epoch - best_epoch > 15:
                break


            tr_x, tr_y = tr_oof[tr_idx], y[tr_idx]
            val_x, val_y = tr_oof[val_idx], y[val_idx]

            val_pred = model.predict(val_x)

            score = label_ranking_average_precision_score(val_y, val_pred)

            if score > best_score:
                best_score = score
                best_epoch = epoch
                oof[val_idx] = val_pred
                model.save_weights(f"../model/stacker{cfg.name}{fold}.h5")

            model.fit(x=tr_x, y=tr_y, batch_size=cfg.bs, verbose=0)
            print(f'{epoch} score {score} ,  best {best_score}...')

        model.load_weights(f"../model/stacker{cfg.name}{fold}.h5")
        predictions += model.predict(test_pred)

    print('lrap: ', label_ranking_average_precision_score(y, oof))
    predictions /= cfg.n_folds
    print(label_ranking_average_precision_score(y,oof))
    test = pd.read_csv('../input/sample_submission.csv')
    test.loc[:, test.columns[1:].tolist()] = predictions
    test.to_csv('submission.csv', index=False)

def average(cfg,files,return_pred = False):
    df = pd.read_csv(f'../input/train_curated.csv')
    y = split_and_label(df['labels'].values)

    result = 0
    oof = 0
    all_w = 0
    for name,w in files.items():
        oof += w * np.load(f'../output/{name}oof.npy')
        print(name,'lrap ',label_ranking_average_precision_score(y,np.load(f'../output/{name}oof.npy')))
        result += w * np.load(f'../output/{name}pred.npy')
        all_w += w

    oof /= all_w
    result /= all_w
    print(label_ranking_average_precision_score(y,oof))
    if return_pred:
        return oof,result
    test = pd.read_csv('../input/sample_submission.csv')
    test.loc[:, test.columns[1:].tolist()] = result
    test.to_csv('submission.csv', index=False)
    # print(test)



if __name__ == '__main__':

    cfg = Config(n_folds=10,lr = 0.0001, batch_size=40)
    # stacking(cfg,{
    #     'model_MSC_se_r4_1.0_10fold_withpretrain_e28_':1.0,
    #     'max3exam':2.1,
    #     'v1mix':2.4,
    #     'model_MSC_se_r4_2.0_10fold_withpretrain_e28_':1.0,
    #     # 'model_se_r4_1.5_10fold_withpretrain_e28_':1.0,
    #     'se_':1,
    #     # 'concat_v1':0,
    #     'se_concat':1,
    #
    # })

    # stacking(cfg, {
    #     'model_MSC_se_r4_1.0_10fold_withpretrain_e28_': 1.0,
    #     'max3exam': 1.9,
    #     'v1mix': 2.1,
    #     'model_MSC_se_r4_2.0_10fold_withpretrain_e28_': 1.0,
    #     'model_se_r4_1.5_10fold_withpretrain_e28_':1.0,
    #     'se_': 0,
    # })

    stacking(cfg, {
        'model_MSC_se_r4_1.0_10fold': 1.0,
        'max3exam': 1.9,
        'v1mix': 2.1,
        'model_MSC_se_r4_2.0_10fold': 1.0,
        'model_se_r4_1.5_10fold': 1.0,
        'se_': 0,
    })
