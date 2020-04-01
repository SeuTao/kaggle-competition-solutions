from include import *
import argparse
from collections import defaultdict
import pandas as pd
from process.data import *
from loss.loss import softmax_loss
from utils import *
from torch.utils.data import DataLoader
from net.archead import *
from loss.cyclic_lr import *
from tqdm import tqdm
from ensemble import *

def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, size_average=False) / num_classes

def get_current_consistency_weight(epoch, consistency, consistency_rampup):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * sigmoid_rampup(epoch, consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def get_model(model, CLASSNUM=1139):
    if  model == 'xception_6channel':
        from net.model_xception_6channel import Net
    elif  model == 'xception_large_6channel':
        from net.model_xception_large_6channel import Net

    net = Net(num_class=CLASSNUM,
              is_arc=config.is_arc,
              arc_s=config.arc_s,
              arc_m=config.arc_m)
    return net

def do_valid(net, valid_loader, predict_num = [0,1108]):
    valid_num  = 0
    truths   = []
    losses   = []
    probs = []
    labels = []

    ids = []
    with torch.no_grad():
        for input, truth_, id in valid_loader:
            ids += id
            input = input.cuda()
            truth_ = truth_.cuda()

            input = to_var(input)
            truth_ = to_var(truth_)

            logit = net.forward(input)
            logit =  logit[:, predict_num[0]: predict_num[1]]
            truth_ = truth_ - predict_num[0]

            loss = softmax_loss(logit, truth_)
            probs.append(logit)
            labels.append(truth_)
            valid_num += len(input)

            loss_tmp = loss.data.cpu().numpy().reshape([1])
            losses.append(loss_tmp)
            truths.append(truth_.data.cpu().numpy())

    assert (valid_num == len(valid_loader.sampler))
    # ------------------------------------------------------
    loss = np.concatenate(losses,axis=0)
    loss = loss.mean()
    prob = torch.cat(probs)
    label = torch.cat(labels)
    _, precision = metric(prob, label)

    print('calculate balance acc')
    logits  = prob.cpu().numpy().reshape([valid_num, 1108])
    prob_balance, _ = balance_plate_probability_training(logits, ids, plate_dict, a_dict, iters=0, is_show = False)
    label_np  = label.cpu().numpy().reshape([valid_num])
    tmp = np.argmax(prob_balance, 1)
    balance_acc = np.mean(label_np == tmp)
    valid_loss = np.array([loss, precision[0], balance_acc, 0.0, prob, label])
    return valid_loss

def run_pretrain(config):

    if config.rgb:
        model = config.model + '_rgb'
    else:
        model = config.model + '_6channel'

    model_name = model+'_'+str(config.image_size)+'_fold'+str(config.train_fold_index)+'_'+config.tag

    base_lr = 3e-3
    config.train_epoch = 160
    def adjust_lr_and_hard_ratio(optimizer, ep):
        if ep < 100:
            lr = 3e-4
            hard_ratio = 1 * 1e-2
        elif ep< 140:
            lr = 1e-4
            hard_ratio = 4 * 1e-3
        else:
            lr = 1e-5
            hard_ratio = 4 * 1e-3

        for p in optimizer.param_groups:
            p['lr'] = lr

        return lr, hard_ratio

    batch_size = config.batch_size
    ## setup  -----------------------------------------------------------------------------
    out_dir = os.path.join(model_save_path, model_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(os.path.join(out_dir,'checkpoint')):
        os.makedirs(os.path.join(out_dir,'checkpoint'))
    if not os.path.exists(os.path.join(out_dir,'train')):
        os.makedirs(os.path.join(out_dir,'train'))
    if not os.path.exists(os.path.join(out_dir,'backup')):
        os.makedirs(os.path.join(out_dir,'backup'))

    if config.pretrained_model is not None:
        initial_checkpoint = os.path.join(out_dir, 'checkpoint', config.pretrained_model)
    else:
        initial_checkpoint = None

    train_dataset = Dataset_(path_data + '/train_fold_'+str(config.train_fold_index)+'.csv',
                             data_dir,
                             mode='train',
                             image_size=config.image_size,
                             rgb=config.rgb,
                             augment=[0,0,0])

    train_loader  = DataLoaderX(train_dataset,
                                shuffle = True,
                                batch_size  = batch_size,
                                drop_last   = True,
                                num_workers = 8,
                                pin_memory  = True)

    valid_dataset = Dataset_(path_data + '/valid_fold_'+str(config.train_fold_index)+'.csv',
                             data_dir,
                             mode='valid',
                             image_size=config.image_size,
                             rgb=config.rgb,
                             augment=[0,0,0])

    valid_loader  = DataLoaderX(valid_dataset,
                                shuffle = False,
                                batch_size  = batch_size,
                                drop_last   = False,
                                num_workers = 8,
                                pin_memory  = True)

    net = get_model(model)

    ## optimiser ----------------------------------
    net = torch.nn.DataParallel(net)
    print(net)
    net = net.cuda()

    log = open(out_dir+'/log.pretrain.txt', mode='a')
    log.write('\t__file__     = %s\n' % __file__)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')
    log.write('\t<additional comments>\n')
    log.write('\t  ... xxx baseline  ... \n')
    log.write('\n')

    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
    assert(len(train_dataset)>=batch_size)
    log.write('batch_size = %d\n'%(batch_size))
    log.write('train_dataset : \n%s\n'%(train_dataset))
    log.write('valid_dataset : \n%s\n'%(valid_dataset))
    log.write('\n')

    ## net ----------------------------------------
    log.write('** net setting **\n')
    if initial_checkpoint is not None:
        log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
        print('\tinitial_checkpoint = %s\n' % initial_checkpoint)

    log.write('%s\n'%(type(net)))
    log.write('\n')

    if config.bias_no_decay:
        print('bias no decay !!!!!!!!!!!!!!!!!!')
        train_params = split_weights(net)
    else:
        train_params = filter(lambda p: p.requires_grad, net.parameters())

    optimizer = torch.optim.Adam(train_params, lr=base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0002)

    iter_smooth = 20
    start_iter = 0

    log.write('\n')
    ## start training here! ##############################################
    log.write('** top_n step 100,60,60,60 **\n')
    log.write('** start training here! **\n')
    log.write('                    |------------ VALID -------------|-------- TRAIN/BATCH ----------|         \n')
    log.write('rate   iter  epoch  | loss   acc-1  acc-5   lb       | loss   acc-1  acc-5   lb      |  time   \n')
    log.write('----------------------------------------------------------------------------------------------------\n')

    print('** start training here! **\n')
    print('                    |------------ VALID -------------|-------- TRAIN/BATCH ----------|         \n')
    print('rate   iter  epoch  | loss   acc-1  acc-5   lb       | loss   acc-1  acc-5   lb      |  time   \n')
    print('----------------------------------------------------------------------------------------------------\n')

    valid_loss   = np.zeros(6,np.float32)
    batch_loss   = np.zeros(6,np.float32)

    i    = 0
    start = timer()
    max_valid = 0

    for epoch in range(config.train_epoch):
        sum_train_loss = np.zeros(6,np.float32)
        sum = 0
        optimizer.zero_grad()

        rate, hard_ratio = adjust_lr_and_hard_ratio(optimizer, epoch + 1)
        print('change lr: '+str(rate))

        for input, truth_, _ in train_loader:
            iter = i + start_iter
            # one iteration update  -------------
            net.train()
            input = input.cuda()
            truth_ = truth_.cuda()

            input = to_var(input)
            truth_ = to_var(truth_)

            if config.is_arc:
                logit = net(input, truth_)
            else:
                logit = net(input)

            loss = softmax_loss(logit, truth_) * config.softmax_w
            _, precision = metric(logit, truth_)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            batch_loss[:4] = np.array((loss.data.cpu().numpy(),
                                       precision[0].data.cpu().numpy(),
                                       precision[2].data.cpu().numpy(),
                                       loss.data.cpu().numpy())).reshape([4])

            sum_train_loss += batch_loss
            sum += 1

            if iter%iter_smooth == 0:
                sum_train_loss = np.zeros(6,np.float32)
                sum = 0

            if i % 10 == 0:
                print(model_name + ' %0.7f %5.1f %6.1f | %0.3f  %0.3f  %0.3f  (%0.3f)%s  | %0.3f  %0.3f  %0.3f  (%0.3f)  | %s' % (\
                             rate, iter, epoch,
                             valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3],' ',
                             batch_loss[0], batch_loss[1], batch_loss[2], batch_loss[3],
                             time_to_str((timer() - start),'min')))

            if i % 100 == 0:
                log.write('%0.7f %5.1f %6.1f | %0.3f  %0.3f  %0.3f  (%0.3f)%s  | %0.3f  %0.3f  %0.3f  (%0.3f)  | %s' % (\
                             rate, iter, epoch,
                             valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3],' ',
                             batch_loss[0], batch_loss[1], batch_loss[2], batch_loss[3],
                             time_to_str((timer() - start),'min')))

                log.write('\n')

            i=i+1
            if i%1000==0:
                net.eval()
                valid_loss = do_valid(net, valid_loader)
                net.train()

                if max_valid < valid_loss[2]:
                    max_valid = valid_loss[2]
                    print('save max valid!!!!!! : ' + str(max_valid))
                    log.write('save max valid!!!!!! : ' + str(max_valid))
                    log.write('\n')
                    torch.save(net.state_dict(), out_dir + '/checkpoint/max_valid_model.pth')

        if (epoch+1) % config.iter_save_interval ==0 and epoch>0:
            torch.save(net.state_dict(), out_dir + '/checkpoint/%08d_model.pth' % (epoch))

        net.eval()
        valid_loss = do_valid(net, valid_loader)
        net.train()

        if max_valid < valid_loss[2]:
            max_valid = valid_loss[2]
            print('save max valid!!!!!! : ' + str(max_valid))
            log.write('save max valid!!!!!! : ' + str(max_valid))
            log.write('\n')
            torch.save(net.state_dict(), out_dir + '/checkpoint/max_valid_model.pth')

def run_finetune_CELL_cyclic_semi_final(config,
                                              cell,
                                              tag = None,
                                              fold_index = 0,
                                              epoch_ratio = 1.0,
                                              online_pseudo = True):

    if config.rgb:
        model = config.model + '_rgb'
    else:
        model = config.model + '_6channel'

    model_name = model + '_' + str(config.image_size)+'_fold'+str(config.train_fold_index)+'_'+config.tag

    batch_size = config.batch_size
    ## setup  -----------------------------------------------------------------------------
    out_dir = os.path.join(model_save_path, model_name)
    print(out_dir)

    if config.pretrained_model is not None:
        initial_checkpoint = os.path.join(out_dir, 'checkpoint', config.pretrained_model)
    else:
        initial_checkpoint = None

    train_dataset = Dataset_(path_data + r'/train_fold_'+str(fold_index)+'.csv',
                                  img_dir=data_dir,
                                  mode='train',
                                  image_size=config.image_size,
                                  rgb=config.rgb,
                                  cell = cell,
                                  augment=[0,0,0],
                                  is_TransTwice=True)

    semi_valid_dataset = Dataset_(path_data + r'/valid_fold_'+str(fold_index)+'.csv',
                                       img_dir=data_dir,
                                       mode='semi_valid',
                                       image_size=config.image_size,
                                       rgb=config.rgb,
                                       cell=cell,
                                       is_TransTwice=True)

    infer_dataset = Dataset_(data_dir + '/test.csv',
                             data_dir,
                             image_size=config.image_size,
                             mode='semi_test',
                             rgb=config.rgb,
                             cell=cell,
                             is_TransTwice=True)

    train_loader  = DataLoader(train_dataset + semi_valid_dataset + infer_dataset,
                                    shuffle = True,
                                    batch_size  = batch_size,
                                    drop_last   = True,
                                    num_workers = 8,
                                    pin_memory  = True)

    pseudo_loader  = DataLoader(semi_valid_dataset + infer_dataset,
                                    shuffle = False,
                                    batch_size  = batch_size,
                                    drop_last   = False,
                                    num_workers = 8,
                                    pin_memory  = True)

    valid_dataset = Dataset_(path_data + r'/valid_fold_'+str(fold_index)+'.csv',
                                  img_dir=data_dir,
                                  mode='valid',
                                  image_size=config.image_size,
                                  rgb=config.rgb,
                                  cell=cell,
                                  augment=[0,0,0])

    valid_loader  = DataLoader(valid_dataset,
                                shuffle = False,
                                batch_size  = batch_size,
                                drop_last   = False,
                                num_workers = 8,
                                pin_memory  = True)

    out_dir = os.path.join(out_dir, 'checkpoint')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if tag is not None:
        cell += '_' + tag

    log = open(out_dir+'/log.'+cell+'_finetune_train.txt', mode='a')
    log.write('\t__file__     = %s\n' % __file__)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')
    log.write('\t<additional comments>\n')
    log.write('\t  ... xxx baseline  ... \n')
    log.write('\n')

    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
    assert(len(train_dataset)>=batch_size)
    log.write('batch_size = %d\n'%(batch_size))
    log.write('train_dataset : \n%s\n'%(train_dataset))
    log.write('valid_dataset : \n%s\n'%(valid_dataset))
    log.write('\n')

    net = get_model(model)
    ema_net = get_model(model)

    for param in ema_net.parameters():
        param.detach_()

    ## net ----------------------------------------
    log.write('** net setting **\n')
    if initial_checkpoint is not None:
        log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_pretrain(initial_checkpoint, is_skip_fc=True)
        ema_net.load_pretrain(initial_checkpoint, is_skip_fc=False)
        print('\tinitial_checkpoint = %s\n' % initial_checkpoint)

    ## optimiser ----------------------------------
    net = torch.nn.DataParallel(net)
    net = net.cuda()
    ema_net = torch.nn.DataParallel(ema_net)
    ema_net = ema_net.cuda()

    log.write('%s\n'%(type(net)))
    log.write('\n')

    if config.bias_no_decay:
        print('bias no decay')
        train_params = split_weights(net)
    else:
        train_params = filter(lambda p: p.requires_grad, net.parameters())

    iter_smooth = 20
    start_iter = 0

    log.write('\n')
    ## start training here! ##############################################
    log.write('** top_n step 100,60,60,60 **\n')
    log.write('** start training here! **\n')
    log.write('                    |------------ VALID -------------|-------- TRAIN/BATCH ----------|         \n')
    log.write('rate   iter  epoch  | loss   acc-1  acc-5   lb       | loss   acc-1  acc-5   lb      |  time   \n')
    log.write('----------------------------------------------------------------------------------------------------\n')

    print('** start training here! **\n')
    print('                    |------------ VALID -------------|-------- TRAIN/BATCH ----------|         \n')
    print('rate   iter  epoch  | loss   acc-1  acc-5   lb       | loss   acc-1  acc-5   lb      |  time   \n')
    print('----------------------------------------------------------------------------------------------------\n')

    ##### pay attention to this
    all_semi_ids =  list(semi_valid_dataset.records['id_code']) + list(infer_dataset.records['id_code'])
    pseudo_logit = np.zeros([len(infer_dataset) + len(semi_valid_dataset), 1108])
    pseudo_labels = np.zeros([len(infer_dataset) + len(semi_valid_dataset), 1])

    def get_pseudo_labels(pseudo_labels, ids, all_semi_ids):
        indexs = [all_semi_ids.index(tmp) for tmp in ids]
        labels = np.argmax(pseudo_labels, axis=1)
        batch_labels = torch.LongTensor(np.asarray(labels[indexs]))
        return batch_labels

    def update_pseudo_labels(pseudo_logit, ids, ratio = 0.9, tta_num = 1, iters = 3000):
        probs_all = np.zeros([len(infer_dataset) + len(valid_dataset), 1108])

        for i in range(tta_num):
            pseudo_num = 0
            probs = []

            with torch.no_grad():
                for _, input, _, _, _ in pseudo_loader:
                    input = input.cuda()
                    input = to_var(input)
                    logit = ema_net.forward(input)
                    logit = logit[:, 0: 1108]
                    probs.append(logit)
                    pseudo_num += len(input)

            assert (pseudo_num == len(pseudo_loader.sampler))
            probs_ = torch.cat(probs).cpu().numpy().reshape([pseudo_num, 1108])
            probs_all += probs_ / tta_num

        pseudo_logit = pseudo_logit * (1-ratio) + probs_all * ratio
        pseudo_labels, ids = balance_plate_probability_training(pseudo_logit, ids, plate_dict, a_dict, iters)
        return pseudo_logit, pseudo_labels

    i    = 0
    start = timer()

    base_lr = 1e-1
    optimizer = torch.optim.SGD(train_params, lr=base_lr, weight_decay=0.0002)

    cycle_inter = int(40 * epoch_ratio)
    cycle_num = 4

    sgdr = CosineAnnealingLR_with_Restart(optimizer,
                                          T_max=cycle_inter,
                                          T_mult=1,
                                          model=net,
                                          out_dir='../input/',
                                          take_snapshot=False,
                                          eta_min=1e-3)

    last_lr = 1e-4
    last_epoch = 5

    epoch_all = 0
    epoch_stop = cycle_inter * cycle_num + last_epoch

    max_valid_all = 0
    max_valid_ema = 0
    valid_loss_ema = np.zeros(6, np.float32)

    for cycle_index in range(cycle_num+1):
        print('cycle index: ' + str(cycle_index))

        max_valid = 0
        valid_loss = np.zeros(6, np.float32)
        batch_loss = np.zeros(6, np.float32)

        for epoch in range(cycle_inter):
            epoch_all += 1

            if epoch_all > epoch_stop:
                return

            elif epoch_all > cycle_inter * cycle_num:
                for p in optimizer.param_groups:
                    p['lr'] = last_lr
                rate = last_lr
            else:
                sgdr.step()
                rate = optimizer.param_groups[0]['lr']

            print('change lr: ' + str(rate))

            net.eval()
            valid_loss = do_valid(net, valid_loader)
            net.train()

            ema_net.eval()
            valid_loss_ema = do_valid(ema_net, valid_loader)
            ema_net.train()

            if max_valid < valid_loss[2] and epoch > 0 :
                max_valid = valid_loss[2]
                print('save max valid!!!!!! : ' + str(max_valid))
                log.write('save max valid!!!!!! : ' + str(max_valid))
                log.write('\n')

            if max_valid_all < valid_loss[2] and epoch > 0 :
                max_valid_all = valid_loss[2]
                print('save max valid all!!!!!! : ' + str(max_valid_all))
                log.write('save max valid all!!!!!! : ' + str(max_valid_all))
                log.write('\n')
                torch.save(net.state_dict(), out_dir + '/max_valid_model_' + cell +
                           '_snapshot_all.pth')

            if epoch_all == (10 * 2 * epoch_ratio - 1):
                iters_balance = 500
                pseudo_logit, pseudo_labels = update_pseudo_labels(pseudo_logit,
                                                                   all_semi_ids,
                                                                   ratio=1.0,
                                                                   tta_num=4,
                                                                   iters=iters_balance)


            if max_valid_ema < valid_loss_ema[2] and epoch > 0 :
                max_valid_ema = valid_loss_ema[2]
                print('save max valid ema!!!!!! : ' + str(max_valid_ema))
                log.write('save max valid ema!!!!!! : ' + str(max_valid_ema))
                log.write('\n')
                torch.save(net.state_dict(), out_dir + '/max_valid_model_' + cell +
                           '_snapshot_ema.pth')

                if online_pseudo:
                    print('update_pseudo_labels')
                    if epoch_all > (10*2*epoch_ratio-1):
                        iters_balance = 500
                        pseudo_logit, pseudo_labels = update_pseudo_labels(pseudo_logit,
                                                                           all_semi_ids,
                                                                           ratio=0.9,
                                                                           tta_num=4,
                                                                           iters=iters_balance)

                    elif epoch_all > (20*2*epoch_ratio-1):
                        iters_balance = 1000
                        pseudo_logit, pseudo_labels = update_pseudo_labels(pseudo_logit,
                                                                           all_semi_ids,
                                                                           ratio=0.9,
                                                                           tta_num=4,
                                                                           iters=iters_balance)

                    elif epoch_all > (40*2*epoch_ratio-1):
                        iters_balance = 3000
                        pseudo_logit, pseudo_labels = update_pseudo_labels(pseudo_logit,
                                                                           all_semi_ids,
                                                                           ratio=0.9,
                                                                           tta_num=8,
                                                                           iters=iters_balance)


            sum_train_loss = np.zeros(6,np.float32)
            sum = 0
            optimizer.zero_grad()

            for input, input_easy, input_hard, truth_, id in train_loader:
                iter = i + start_iter

                # one iteration update  -------------
                net.train()
                input = input.cuda()
                input_easy = input_easy.cuda()
                input_hard = input_hard.cuda()
                truth_ = truth_.cuda()

                input = to_var(input)
                input_easy = to_var(input_easy)
                input_hard = to_var(input_hard)
                truth_ = to_var(truth_)

                indexs_supervised = (truth_ != -1).nonzero().view(-1)
                indexs_semi = (truth_ == -1).nonzero().view(-1)

                if len(indexs_semi) == 0 or len(indexs_supervised) == 0 :
                    continue

                if config.is_arc:
                    logit_arc = net(input[indexs_supervised], truth_[indexs_supervised])
                    logit = net(input_easy)
                    ema_logit = ema_net(input_hard)

                ##################### consistency loss ########################################################################
                consistency = 100.0
                consistency_rp = 5

                if consistency:
                    consistency_weight = get_current_consistency_weight(epoch_all, consistency, consistency_rp)
                    ema_logit = Variable(ema_logit.detach().data, requires_grad=False)
                    consistency_loss = consistency_weight * softmax_mse_loss(logit, ema_logit) / batch_size
                else:
                    consistency_loss = 0

                ##################### online_pseudo label #####################################################################
                if online_pseudo:
                    if epoch_all < (10*2*epoch_ratio):
                        weight = 0.0
                    else:
                        weight = 0.05

                    id_smi = [id[index] for index in list(indexs_semi.cpu().numpy().reshape([-1]))]
                    id_smi_pseudo = get_pseudo_labels(pseudo_labels, id_smi, all_semi_ids).cuda()
                    id_smi_pseudo = to_var(id_smi_pseudo)

                    pseudo_loss = softmax_loss(logit[indexs_semi], id_smi_pseudo) * weight
                    pseudo_loss_log = pseudo_loss.data.cpu().numpy()
                    supervised_loss = softmax_loss(logit_arc, truth_[indexs_supervised]) * (1.0 - weight)

                else:
                    pseudo_loss = 0.0
                    pseudo_loss_log = 0.0
                    supervised_loss = softmax_loss(logit_arc, truth_[indexs_supervised])

                ##################### loss ####################################################################################
                loss = supervised_loss + consistency_loss + pseudo_loss

                _, precision = metric(logit_arc, truth_[indexs_supervised])

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                batch_loss[:4] = np.array((precision[0].data.cpu().numpy(),
                                           supervised_loss.data.cpu().numpy(),
                                           pseudo_loss_log,
                                           consistency_loss.data.cpu().numpy())).reshape([4])

                sum_train_loss += batch_loss
                sum += 1

                if epoch_all > 20:
                    alpha_ema = 0.999
                elif epoch_all > 10:
                    alpha_ema = 0.99
                elif epoch_all > 1:
                    alpha_ema = 0.9
                else:
                    alpha_ema = 0.5

                update_ema_variables(net, ema_net, alpha_ema, i)

                if iter%iter_smooth == 0:
                    sum_train_loss = np.zeros(6,np.float32)
                    sum = 0

                if i % 10 == 0:
                    print(model_name +' finetune '+ cell +'%6.1f %0.7f %5.1f %6.1f | %0.3f  %0.3f  %0.3f  (%0.3f)%s  | %0.3f  %0.3f  %0.3f  (%0.3f)%s  | %0.3f  %0.3f  %0.3f  (%0.3f)  | %s' % (\
                                 cycle_index, rate, iter, epoch,
                                 valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3],' ',
                                 valid_loss_ema[0], valid_loss_ema[1], valid_loss_ema[2], valid_loss_ema[3], ' ',
                                 batch_loss[0], batch_loss[1], batch_loss[2], batch_loss[3],
                                 time_to_str((timer() - start),'min')))



                if i % 100 == 0:
                    log.write('%6.1f %0.7f %5.1f %6.1f | %0.3f  %0.3f  %0.3f  (%0.3f)%s  |%0.3f  %0.3f  %0.3f  (%0.3f)%s  | %0.3f  %0.3f  %0.3f  (%0.3f)  | %s' % (\
                                 cycle_index, rate, iter, epoch,
                                 valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3],' ',
                                 valid_loss_ema[0], valid_loss_ema[1], valid_loss_ema[2], valid_loss_ema[3], ' ',
                                 batch_loss[0], batch_loss[1], batch_loss[2], batch_loss[3],
                                 time_to_str((timer() - start),'min')))

                    log.write('\n')

                i=i+1

def run_infer_oof_finetuned(config, cell, initial_checkpoint):

    if config.rgb:
        model = config.model + '_rgb'
    else:
        model = config.model + '_6channel'

    batch_size = config.batch_size
    ## setup  -----------------------------------------------------------------------------
    net = get_model(model)
    net = torch.nn.DataParallel(net)

    if initial_checkpoint is not None:
        print(initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

    net = net.cuda()
    net.eval()

    augments = []
    # 8TTA
    augments.append([0, 0, 0])
    augments.append([0, 0, 1])
    augments.append([0, 1, 0])
    augments.append([0, 1, 1])
    augments.append([1, 0, 0])
    augments.append([1, 0, 1])
    augments.append([1, 1, 0])
    augments.append([1, 1, 1])

    predict_num = 1108

    def get_valid_label(fold=0):
        df = pd.read_csv(path_data + '/valid_fold_' + str(fold) + '.csv')
        if cell is not None:
            df_all = []
            for i in range(100):
                df_ = pd.DataFrame(df.loc[df['experiment'] == (cell + '-' + str(100 + i)[-2:])])
                df_all.append(df_)
            df = pd.concat(df_all)

        val_label = np.asarray(list(df[r'sirna'])).reshape([-1])
        return val_label, list(df[r'id_code'])

    val_label, ids = get_valid_label(fold=config.train_fold_index)

    #####################################infer valid#############################################
    probs_all = []
    for index in range(len(augments)):
        for site in range(2):

            valid_dataset = Dataset_(path_data + '/valid_fold_' + str(config.train_fold_index) + '.csv',
                                     data_dir,
                                     mode='valid',
                                     image_size=config.image_size,
                                     site=site+1,
                                     rgb=config.rgb,
                                     cell=cell,
                                     augment=augments[index])

            valid_loader = DataLoaderX(valid_dataset,
                                      shuffle=False,
                                      batch_size=batch_size,
                                      drop_last=False,
                                      num_workers=4,
                                      pin_memory=True)

            # infer test
            test_ids = []
            probs = []
            from tqdm import tqdm
            for i,(input, label, id) in enumerate(tqdm(valid_loader)):
                test_ids += id
                input = input.cuda()
                input = to_var(input)
                logit = net.forward(input)
                logit = logit[:, 0:predict_num]

                prob = logit
                probs += prob.data.cpu().numpy().tolist()

            probs = np.asarray(probs)

            val_ = np.argmax(probs, axis=1).reshape([-1])
            print(np.mean(val_ == val_label))

            probs_all.append(probs)

    probs_all = np.mean(probs_all,axis=0)
    print(probs_all.shape)

    val_ = np.argmax(probs_all, axis=1).reshape([-1])
    print(np.mean(val_ == val_label))

    save_path = initial_checkpoint.replace('.pth', '_npy')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_path = os.path.join(save_path,  cell+'_val.npy')
    print(save_path)
    np.save(save_path, probs_all)

    # return
    #####################################infer test#############################################
    probs_all = []
    for index in range(len(augments)):
        for site in range(2):
            valid_dataset = Dataset_(data_dir+'/test.csv',
                                     data_dir,
                                     mode='test',
                                     image_size=config.image_size,
                                     site=site+1,
                                     rgb=config.rgb,
                                     cell=cell,
                                     augment=augments[index])

            valid_loader = DataLoaderX(valid_dataset,
                                      shuffle=False,
                                      batch_size=batch_size,
                                      drop_last=False,
                                      num_workers=4,
                                      pin_memory=True)

            # infer test
            test_ids = []
            probs = []
            from tqdm import tqdm
            for i,(id, input) in enumerate(tqdm(valid_loader)):
                test_ids += id
                input = input.cuda()
                input = to_var(input)
                logit = net.forward(input)
                logit = logit[:, 0:predict_num]

                prob = logit
                probs += prob.data.cpu().numpy().tolist()

            probs = np.asarray(probs)
            probs_all.append(probs)

    probs_all = np.mean(probs_all,axis=0)
    print(probs_all.shape)

    save_path = initial_checkpoint.replace('.pth', '_npy')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_path = os.path.join(save_path,  cell + '_test.npy')
    print(save_path)
    np.save(save_path, probs_all)

def main(config):

    if config.mode == 'pretrain':
        # pretrain on all cell types
        run_pretrain(config)


    elif config.mode == 'semi_finetune':
        # semi-supervised learning finetune on each cell type
        cell_types = ['U2OS','RPE','HEPG2','HUVEC']
        epoch_ratios = [ 1.0, 0.5, 0.5, 0.25]

        for cell, epoch_ratio in zip(cell_types, epoch_ratios):
            config.pretrained_model = r'max_valid_model.pth'
            config.finetune_tag = r'semi'
            run_finetune_CELL_cyclic_semi_final(config,
                                                cell = cell,
                                                fold_index = config.train_fold_index,
                                                tag = config.finetune_tag,
                                                epoch_ratio = epoch_ratio,
                                                online_pseudo = True)


    elif config.mode == 'infer':

        if config.rgb:
            model = config.model + '_rgb'
        else:
            model = config.model + '_6channel'

        cell_types = ['U2OS','RPE','HEPG2','HUVEC']
        for cell in cell_types:
            model_name = model + '_' + str(config.image_size) + '_fold' + str(config.train_fold_index) + '_' + config.tag
            out_dir = os.path.join(model_save_path, model_name)
            initial_checkpoint = os.path.join(out_dir, 'checkpoint', 'max_valid_model_' + cell + '_semi_snapshot_all.pth')

            with torch.no_grad():
                run_infer_oof_finetuned(config, cell=cell, initial_checkpoint=initial_checkpoint)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type=str, default='final')
    parser.add_argument('--train_fold_index', type=int, default = 0)
    parser.add_argument('--rgb', type=bool, default=False)
    parser.add_argument('--is_arc', type=bool, default=True)
    parser.add_argument('--arc_s', type=float, default=30.0)
    parser.add_argument('--arc_m', type=float, default=0.1)
    parser.add_argument('--bias_no_decay', type=bool, default=False)

    parser.add_argument('--model', type=str, default='xception_large')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--softmax_w', type=float, default=1.0)

    parser.add_argument('--mode', type=str, default='pretrain', choices=['pretrain', 'semi_finetune', 'infer'])
    parser.add_argument('--pretrained_model', type=str, default=None)

    parser.add_argument('--iter_save_interval', type=int, default=10)
    parser.add_argument('--train_epoch', type=int, default=80)

    config = parser.parse_args()
    print(config)
    main(config)



