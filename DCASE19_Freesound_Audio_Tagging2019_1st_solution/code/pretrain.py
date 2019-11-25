from tqdm import tqdm
from sklearn.metrics import label_ranking_average_precision_score
from utils import *
from config import *


def main(cfg,get_model):

    if True: # load data
        df = pd.read_csv(f'../input/train_noisy.csv')
        y = split_and_label(df['labels'].values)
        x = train_dir + df['fname'].values
        x = [librosa.load(path, 44100)[0] for path in tqdm(x)]
        x = [librosa.effects.trim(data)[0] for data in tqdm(x)]

        gfeat = np.load('../input/gfeat.npy')[-len(x):]


        df = pd.read_csv(f'../input/train_curated.csv')
        val_y = split_and_label(df['labels'].values)
        val_x = train_dir + df['fname'].values
        val_x = [librosa.load(path, 44100)[0] for path in tqdm(val_x)]
        val_x = [librosa.effects.trim(data)[0] for data in tqdm(val_x)]
        val_gfeat = np.load('../input/gfeat.npy')[:len(val_x)]

    print(cfg)

    if True: # init
        K.clear_session()
        model = get_model(cfg)
        best_score = -np.inf

    for epoch in range(35):

        if epoch in cfg.milestones:
            K.set_value(model.optimizer.lr, K.get_value(model.optimizer.lr) * cfg.gamma)

        tr_loader = FreeSound(x, gfeat, y, cfg, 'train', epoch)
        val_loaders = [FreeSound(val_x, val_gfeat, val_y, cfg, f'pred{i+1}', epoch) for i in range(3)]

        model.fit_generator(
            tr_loader,
            steps_per_epoch=len(tr_loader),
            verbose=0,
            workers=6
        )
        val_pred = [model.predict_generator(vl, workers=4) for vl in val_loaders]
        ave_val_pred = np.average(val_pred, axis=0)
        score = label_ranking_average_precision_score(val_y, ave_val_pred)

        if epoch >= 28 and score > best_score:
            best_score = score
            model.save_weights(f"../model/{cfg.name}pretrainedbest.h5")

        if epoch >= 28:
            model.save_weights(f"../model/{cfg.name}pretrained{epoch}.h5")
            print(f'{epoch} score {score},  best {best_score}...')





if __name__ == '__main__':
    from models import *

    cfg = Config(
        duration=5,
        name='v1mix',
        lr=0.0005,
        batch_size=32,
        rnn_unit=128,
        momentum=0.85,
        mixup_prob=0.7,
        lm=0.01,
        pool_mode=('max', 'avemax1'),
        x1_rate=0.7,
        milestones=(8,12,16),
        get_backbone=get_conv_backbone
    )
    main(cfg, cnn_model)

    cfg = Config(
        duration=5,
        name='model_MSC_se_r4_1.0_10fold',
        lr=0.0005,
        batch_size=32,
        rnn_unit=128,
        momentum=0.85,
        mixup_prob=0.7,
        lm=0.01,
        pool_mode=('max', 'avemax1'),
        x1_rate=0.7,
        milestones=(8, 12, 16),
        get_backbone=model_se_MSC,
        w_ratio=1,
    )
    main(cfg, cnn_model)

    cfg = Config(
        duration=5,
        name='model_MSC_se_r4_2.0_10fold',
        lr=0.0005,
        batch_size=32,
        rnn_unit=128,
        momentum=0.85,
        mixup_prob=0.7,
        lm=0.01,
        pool_mode=('max', 'avemax1'),
        x1_rate=0.7,
        milestones=(8, 12, 16),
        get_backbone=model_se_MSC,
        w_ratio=2.0,
    )
    main(cfg, cnn_model)

    cfg = Config(
        duration=5,
        name='model_se_r4_1.5_10fold',
        lr=0.0005,
        batch_size=32,
        rnn_unit=128,
        momentum=0.85,
        mixup_prob=0.7,
        lm=0.01,
        pool_mode=('max', 'avemax1'),
        x1_rate=0.7,
        milestones=(8, 12, 16),
        get_backbone=model_se_MSC,
        w_ratio=1.5,
    )
    main(cfg, cnn_model)

    cfg = Config(
        duration=5,
        name='se',
        lr=0.0005,
        batch_size=32,
        rnn_unit=128,
        momentum=0.85,
        mixup_prob=0.7,
        lm=0.01,
        pool_mode=('max', 'avemax1'),
        x1_rate=0.7,
        milestones=(8, 12, 16),
        get_backbone=get_se_backbone
    )
    main(cfg, cnn_model)





