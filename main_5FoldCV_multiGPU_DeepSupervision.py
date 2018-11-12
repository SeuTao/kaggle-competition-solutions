import argparse
from data_loader import *
from torch.backends import cudnn
from torchvision.utils import save_image
from model.model_stnet import StNetV2, StNetV2_slim, \
                              model34_DeepSupervised, \
                              StNetV2_slim_DeepSupervised, \
                              StNetV2_slim_DeepSupervised_slim,\
                              model50_DeepSupervised,\
                              StNetV2_slim_DeepSupervised_Dblock,\
                              StNetV2_slim_DeepSupervised_Dblock_SCSE


from torch.nn import functional as F
from evaluation.metrics import *
from utils import create_submission
from loss.loss import *
import time
import datetime
import cv2
from metric import f2

class DecoderSolver(object):
    def __init__(self, config):
        self.model_name = config.model_name
        self.lr_steps = config.lr_steps.split(',')
        self.lr_steps = [int(tmp) for tmp in self.lr_steps]

        self.fmap = config.fmap
        self.model = config.model
        self.is_freeze = config.is_freeze
        self.is_bottleneck = config.is_bottleneck

        # Model hyper-parameters
        self.image_size = config.image_size

        # Hyper-parameteres
        self.g_lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        self.dice_weight = config.dice_weight
        self.bce_weight = config.bce_weight

        # Training settings
        self.num_epochs = config.num_epochs
        self.num_iters = config.num_iters
        # self.num_iters_decay = config.num_iters_decay
        self.batch_size = config.batch_size
        self.pretrained_model = config.pretrained_model

        # Test settings
        self.test_model = config.test_model

        # Path
        self.log_path = os.path.join('./models', self.model_name, config.log_path)
        self.sample_path = os.path.join('./models', self.model_name, config.sample_path)
        self.model_save_path = os.path.join('./models', self.model_name, config.model_save_path)
        self.result_path = os.path.join('./models', self.model_name, config.result_path)

        # Create directories if not exist
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        if not os.path.exists(self.sample_path):
            os.makedirs(self.sample_path)
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        # Step size
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step

        # Build tensorboard if use
        self.build_model()

        def load_cvpr_model():
            CVPR_pretrained_model = r'/data2/shentao/Projects/Kaggle_airbus/models/CVPR_pretraind_model/dink34.pth'
            CVPR_state_dict = torch.load(CVPR_pretrained_model)

            state_dict = self.G.state_dict()
            keys = list(CVPR_state_dict.keys())
            for key in keys:
                if  ('finalconv3' in key):
                    print(key + ' excluded!!!!!!!!!!!!!!!')
                    continue
                print(key + ' loaded')

                new_key = key.replace(r'module.',r'')
                print(new_key)
                state_dict[new_key] = CVPR_state_dict[key]

            self.G.load_state_dict(state_dict)
            for param in self.G.state_dict():
                print(param)
            print('LOAD initial model!!!!!!!!')

        self.load_pretrained_model(config.train_fold_index)

    def build_model(self):
        if self.model == 'stnet34_V2':
            self.G = StNetV2(num_classes=1, num_filters=self.fmap, pretrained=True, is_deconv=True, is_Refine=False,
                             is_Freeze=self.is_freeze, is_SCSEBlock = False, is_bottleneck=self.is_bottleneck, norm_type='batch_norm')

        elif self.model == 'stnet34_V2_slim':
            self.G = StNetV2_slim(num_classes=1, num_filters=self.fmap)

        elif self.model == 'model34_DeepSupervised':
            self.G = StNetV2_slim_DeepSupervised()

        elif self.model == 'model34_DeepSupervised_Dblock':
            self.G = StNetV2_slim_DeepSupervised_Dblock()

        elif self.model == 'model50_DeepSupervised':
            self.G = model50_DeepSupervised()

        elif self.model == 'model34_DeepSupervised_slim':
            self.G = StNetV2_slim_DeepSupervised_slim()

        elif self.model == 'model34_DeepSupervised_Dblock_SCSE':
            self.G = StNetV2_slim_DeepSupervised_Dblock_SCSE()


        self.g_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.G.parameters()),
                                           self.g_lr, weight_decay=0.0002, momentum=0.9)
        self.print_network(self.G, 'G')

        if torch.cuda.is_available():
            self.G = torch.nn.DataParallel(self.G)
            self.G.cuda()

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def load_pretrained_model(self, fold_index):
        if os.path.exists(os.path.join(self.model_save_path, 'fold_' + str(fold_index),
                                       '{}_G.pth'.format(self.pretrained_model))):

            g_dict = self.G.state_dict()
            dict = torch.load(os.path.join(self.model_save_path, 'fold_' + str(fold_index),
                                    '{}_G.pth'.format(self.pretrained_model)))

            for id in g_dict:
                if id in dict:
                    # print('load')
                    # print(id)
                    g_dict[id] = dict[id]
                else:
                    print('except')
                    print(id)

            self.G.load_state_dict(g_dict)
            print('loaded trained G models fold: {} (step: {})..!'.format(fold_index, self.pretrained_model))

    def update_lr(self, g_lr):
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr

    def to_var(self, x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile)

    def denorm_image(self, x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def denorm_output(self, x):
        out = x
        return out.clamp_(0, 1)

    def threshold(self, x):
        x = x.clone()
        x[x >= 0.5] = 1
        x[x < 0.5] = 0
        return x

    def train_fold(self, fold_index):
        CE = torch.nn.CrossEntropyLoss()
        if not os.path.exists(os.path.join(self.model_save_path,'fold_'+str(fold_index))):
            os.makedirs(os.path.join(self.model_save_path,'fold_'+str(fold_index)))

        print('train loader!!!')
        data_loader = get_5foldloader(self.image_size, self.batch_size, fold_index, mode= 'train')

        val_loader = get_5foldloader(self.image_size, self.batch_size, fold_index, mode = 'val')

        for i, (images, labels, _) in enumerate(data_loader):
            fixed_images = self.to_var(images, volatile=True)
            fixed_labels = self.to_var(labels, volatile=True)
            break

        save_fix_image = self.denorm_image(fixed_images.data)
        save_fix_label = self.denorm_output(fixed_labels.data)

        iters_per_epoch = len(data_loader)
        lr_tmp = self.g_lr
        # Start training
        start_time = time.time()
        for init_index in range(self.num_epochs):
            for i, (images, labels, is_empty) in enumerate(data_loader):
                inputs = self.to_var(images)
                labels = self.to_var(labels)

                class_lbls = self.to_var(torch.LongTensor(is_empty))
                binary_logits, no_empty_logits, _, final_logits, _ = self.G.forward(inputs)

                bce_loss_final = mixed_dice_bce_loss(final_logits, labels, dice_weight=self.dice_weight, bce_weight=self.bce_weight)
                class_loss = CE(binary_logits, class_lbls)

                non_empty = []
                for c in range(len(is_empty)):
                    if is_empty[c] == 0:
                        non_empty.append(c)

                has_empty_nonempty = False
                if len(non_empty) * len(is_empty) > 0:
                    has_empty_nonempty = True

                all_loss = bce_loss_final + 0.05 * class_loss

                loss = {}
                loss['loss_seg'] = bce_loss_final.data[0]
                loss['loss_classifier'] = class_loss.data[0]

                if has_empty_nonempty:
                    indices = self.to_var(torch.LongTensor(non_empty))
                    y_non_empty = torch.index_select(no_empty_logits, 0, indices)
                    mask_non_empty = torch.index_select(labels, 0, indices)
                    loss_no_empty = mixed_dice_bce_loss(y_non_empty, mask_non_empty, dice_weight=self.dice_weight, bce_weight=self.bce_weight)
                    all_loss += 0.50 * loss_no_empty
                    loss['loss_seg_noempty'] = loss_no_empty.data[0]

                self.g_optimizer.zero_grad()
                all_loss.backward()
                self.g_optimizer.step()

                # Print out log info
                if (i + 1) % self.log_step == 0:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))

                    lr = self.g_optimizer.param_groups[0]['lr']
                    log = "{} FOLD: {},Elapsed [{}], Epoch [{}/{}],Iter [{}/{}], lr {:.4f}".format(
                        self.model_name, fold_index, elapsed, init_index, self.num_epochs,i+1,iters_per_epoch, lr)
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)


                if (i + 1) % 1000 == 0:
                    fixed_output = self.G(fixed_images)
                    save_fix_out = self.denorm_output(fixed_output[4].data)
                    # save_fix_out = F.upsample(save_fix_out, scale_factor=768, mode='bilinear')
                    save = torch.cat([save_fix_label, save_fix_out], dim=3)
                    save_image(save, os.path.join(self.sample_path, '{}_{}_output.png'.format(init_index + 1, i + 1)),
                               nrow=1, padding=0)
                    print('Translated images and saved into {}..!'.format(self.sample_path))

                    self.val_NoTTA(0, is_load=False, val_loader=val_loader)

            # Decay learning rate
            if (init_index+1) in self.lr_steps:
                lr_tmp *= 0.1
                self.update_lr(lr_tmp)
                print ('Decay learning rate to g_lr: {}'.format(lr_tmp))

            if (init_index + 1) % self.sample_step == 0:
                torch.save(self.G.state_dict(),
                           os.path.join(self.model_save_path, 'fold_' + str(fold_index),
                          '{}_final_G.pth'.format(init_index + 1)))

            # self.val_NoTTA(0, is_load=False, val_loader=val_loader)

    def val(self, fold_index, is_load, is_minAreaRect = True):
        if is_load:
            self.load_pretrained_model(fold_index)

        self.G.eval()
        val_dir = r'/data/shentao/Airbus/AirbusShipDetectionChallenge/train_v2'
        val_loader = get_5foldloader(self.image_size, 1, fold_index, mode = 'val')

        iou_list = []
        for i, (image_path, images, labels, _) in enumerate(val_loader):
                labels = self.to_var(labels)
                label_mat = labels.data
                label_mat = label_mat.cpu().numpy()

                image_path = os.path.join(val_dir,image_path[0])
                output_mat = self.infer_one_img_from_path_8(image_path)

                # print(output_mat.shape)
                output_mat = output_mat.reshape([self.image_size,self.image_size])
                output_mat = cv2.resize(output_mat, (768, 768))
                label_mat = label_mat.reshape([768, 768])

                label_mat[label_mat > 0.5] = 1
                label_mat[label_mat <= 0.5] = 0
                output_mat[output_mat > 0.5] = 1
                output_mat[output_mat <= 0.5] = 0

                label_mat = label_mat.astype(np.uint8)
                output_mat = output_mat.astype(np.uint8)

                if is_minAreaRect:
                    box_mask, contours, _ = cv2.findContours(output_mat, mode=cv2.RETR_EXTERNAL,
                                                             method=cv2.CHAIN_APPROX_NONE)
                    for i in range(len(contours)):
                        bounding_box = cv2.minAreaRect(contours[i])
                        bounding_box = cv2.boxPoints(bounding_box)
                        bounding_box = np.array(bounding_box, dtype=np.int32)

                        p0 = bounding_box[0]
                        p1 = bounding_box[1]
                        p2 = bounding_box[2]

                        dist1 = np.sqrt((p0[0] - p1[0]) * (p0[0] - p1[0]) + (p0[1] - p1[1]) * (p0[1] - p1[1]))
                        dist2 = np.sqrt((p2[0] - p1[0]) * (p2[0] - p1[0]) + (p2[1] - p1[1]) * (p2[1] - p1[1]))

                        s = dist1 * dist2
                        thres_up = 10
                        thres_down = 10

                        if s < thres_down:
                            bounding_box = bounding_box.reshape([1, bounding_box.shape[0], bounding_box.shape[1]])
                            cv2.fillPoly(output_mat, bounding_box, 0)
                        elif s < thres_up:
                            print(str(s) + ' ' + str(thres_up))
                            bounding_box = bounding_box.reshape([1, bounding_box.shape[0], bounding_box.shape[1]])
                            cv2.fillPoly(output_mat, bounding_box, 1)

                f2_tmp = f2([label_mat], [output_mat])
                iou_list.append(f2_tmp)

        Valid_F2 =  sum(iou_list)/(1.0*len(iou_list))
        print('Valid F2: %.3f'%(Valid_F2))
        self.G.train()
        return Valid_F2

    def val_NoTTA(self, fold_index, is_load, val_loader, is_minAreaRect = True):
        if is_load:
            self.load_pretrained_model(fold_index)
        self.G.eval()


        F2_list_no_ship = []
        F2_list_1_ship = []
        F2_list_2_4_ship = []
        F2_list_5_9_ship = []
        F2_list_10_ship = []


        wrong_num = 0

        c = 0
        for i, ( _, images, labels, _) in enumerate(val_loader):
                labels = self.to_var(labels)
                images = self.to_var(images)

                # images_numpy = images.data.cpu().numpy()
                label_mat = labels.data
                label_mat = label_mat.cpu().numpy()

                output_mat = self.G.forward(images)[4].data.cpu().numpy()

                label_mat = label_mat.reshape([-1, 768, 768])
                output_mat = output_mat.reshape([-1,self.image_size,self.image_size])

                for sample_i in range(label_mat.shape[0]):
                    label_tmp = label_mat[sample_i]
                    output_tmp = output_mat[sample_i]

                    # images_numpy_tmp = images_numpy[sample_i]
                    # images_numpy_tmp = np.transpose(images_numpy_tmp,(1,2,0))
                    # print(output_tmp.shape)

                    output_tmp = cv2.resize(output_tmp, (768, 768))

                    thres = 0.5
                    label_tmp[label_tmp > thres] = 1
                    label_tmp[label_tmp <= thres] = 0
                    output_tmp[output_tmp > thres] = 1
                    output_tmp[output_tmp <= thres] = 0
                #
                    label_tmp = label_tmp.astype(np.uint8)
                    output_tmp = output_tmp.astype(np.uint8)

                    # if is_minAreaRect:
                    #     box_mask, contours, _ = cv2.findContours(output_tmp, mode=cv2.RETR_EXTERNAL,
                    #                                              method=cv2.CHAIN_APPROX_NONE)
                    #     for i in range(len(contours)):
                    #         bounding_box = cv2.minAreaRect(contours[i])
                    #         bounding_box = cv2.boxPoints(bounding_box)
                    #         bounding_box = np.array(bounding_box, dtype=np.int32)
                    #
                    #         p0 = bounding_box[0]
                    #         p1 = bounding_box[1]
                    #         p2 = bounding_box[2]
                    #
                    #         dist1 = np.sqrt((p0[0]-p1[0])*(p0[0]-p1[0]) + (p0[1]-p1[1])*(p0[1]-p1[1]))
                    #         dist2 = np.sqrt((p2[0] - p1[0]) * (p2[0] - p1[0]) + (p2[1] - p1[1]) * (p2[1] - p1[1]))
                    #
                    #         s = dist1*dist2
                    #         thres_up = 10
                    #         thres_down = 10
                    #
                    #         if s<thres_down:
                    #             bounding_box = bounding_box.reshape([1, bounding_box.shape[0], bounding_box.shape[1]])
                    #             cv2.fillPoly(output_tmp, bounding_box, 0)
                    #         elif s<thres_up:
                    #             print(str(s)+ ' ' + str(thres_up))
                    #             bounding_box = bounding_box.reshape([1, bounding_box.shape[0], bounding_box.shape[1]])
                    #             cv2.fillPoly(output_tmp, bounding_box, 1)

                    # cv2.imwrite('./tmp/tmp'+str(c)+'.png', output_tmp*255)
                    # cv2.imwrite('./tmp/tmp_img' + str(c) + '.png', np.uint8((images_numpy_tmp + 1)/2.0*255.0))

                    c += 1

                    if np.sum(label_tmp) == 0:
                        F2_list_no_ship.append(f2([label_tmp], [output_tmp]))

                        if np.sum(output_tmp) != 0:
                            wrong_num += 1

                    else:
                        ret, labels = cv2.connectedComponents(label_tmp, connectivity=8)
                        ship_num = np.max(labels)

                        if ship_num == 1:
                            F2_list_1_ship.append(f2([label_tmp],[output_tmp]))
                        elif ship_num <=4:
                            F2_list_2_4_ship.append(f2([label_tmp], [output_tmp]))
                        elif ship_num <=9:
                            F2_list_5_9_ship.append(f2([label_tmp], [output_tmp]))
                        else:
                            F2_list_10_ship.append(f2([label_tmp], [output_tmp]))

                        if np.sum(output_mat) == 0:
                            wrong_num += 1


        F2 = F2_list_no_ship + F2_list_1_ship+F2_list_2_4_ship+F2_list_5_9_ship+F2_list_10_ship
        Valid_F2 =  sum(F2)/(1.0*len(F2))
        print('Valid F2: %.3f'%(Valid_F2))

        acc = 1 - wrong_num / (1.0*len(F2))
        print('acc : %.3f'%(acc))

        F2 = F2_list_1_ship+F2_list_2_4_ship+F2_list_5_9_ship+F2_list_10_ship
        Valid_F2 =  sum(F2)/(1.0*len(F2))
        print('with ship Valid F2: %.3f'%(Valid_F2))

        Valid_F2 =  sum(F2_list_1_ship)/(1.0*len(F2_list_1_ship))
        print('1 ship Valid F2: %.3f'%(Valid_F2))
        Valid_F2 =  sum(F2_list_2_4_ship)/(1.0*len(F2_list_2_4_ship))
        print('2-4 ship Valid F2: %.3f'%(Valid_F2))
        Valid_F2 =  sum(F2_list_5_9_ship)/(1.0*len(F2_list_5_9_ship))
        print('5-9 ship Valid F2: %.3f'%(Valid_F2))
        Valid_F2 =  sum(F2_list_10_ship)/(1.0*len(F2_list_10_ship))
        print('10 ship Valid F2: %.3f'%(Valid_F2))

        self.G.train()
        return 0.0

    def get_label_dict(self, path):
        f = open(path, 'r')

        lines = f.readlines()
        dict = {}
        for line in lines:
            line = line.strip().split(' ')
            id = line[0]
            label = line[1]
            dict[id] = label

        return dict

    def infer_one_img_from_path_8(self, path):
        img = cv2.imread(path,1)

        if img is None:
            img = np.zeros([self.image_size,self.image_size,3])

        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img.reshape([1,self.image_size,self.image_size,3])
        img = img.transpose(0, 3, 1, 2)

        img1 = np.array(img)
        img2 = np.array(img1)[:, :, ::-1, :]
        img3 = np.array(img1)[:, :, :, ::-1]
        img4 = np.array(img2)[:, :, :, ::-1]

        img_flip = np.concatenate([img1, img2, img3, img4])
        img_flip_trans = np.transpose(img_flip, (0,1,3,2))

        img_all = np.concatenate([img_flip, img_flip_trans],axis=0)
        img_all = self.to_var(torch.FloatTensor(img_all))
        img_all = (img_all - 127.5) / 127.5
        img_all = self.G.forward(img_all)

        # img_all_out_nonempty = img_all[2].data.cpu().numpy()
        img_all_out_all = img_all[4].data.cpu().numpy()
        img_all = img_all_out_all

        mask0 = img_all[0:4]
        mask1 = img_all[4:8].transpose(0,1,3,2)

        mask = mask0 + mask1
        mask = mask[0:1] + mask[1:2][:, :, ::-1, :] + mask[2:3][:, :, :, ::-1] + mask[3:4][:, :, ::-1, ::-1]
        return mask / 8.0

    def infer_old(self, fold_index):
        npy_save_path = os.path.join(self.model_save_path, 'fold_' + str(fold_index), self.pretrained_model+'_final_output')
        if not os.path.exists(npy_save_path):
            os.makedirs(npy_save_path)

        self.load_pretrained_model(fold_index)
        self.G.eval()

        test_dir = r'/data/shentao/Airbus/AirbusShipDetectionChallenge/test_v2'
        test_loader = get_5foldloader(self.image_size, 1, 0, mode='test')

        for i, (id) in enumerate(test_loader):
            image_path = os.path.join(test_dir, id[0])
            save_npy = os.path.join(npy_save_path, id[0])

            if not os.path.exists(save_npy.replace(r'.jpg',r'.jpg.npy')):
                output_mat = self.infer_one_img_from_path_8(image_path)
                output_mat = output_mat.reshape([self.image_size, self.image_size])
                output_mat[output_mat > 0.5] = 1
                output_mat[output_mat <= 0.5] = 0
                output_mat = output_mat.astype(np.uint8)
                np.save(os.path.join(npy_save_path, id[0]), output_mat)
            # else:
            #     print(save_npy)

            if i % 1000 == 0 and i > 0:
                print(i)

    def infer(self, fold_index):

        from Ensemble import get_empty_dict

        empty_path = r'/data/shentao/Airbus/code/models/model34_DeepSupervised_Dblock_resize768/models/fold_0/12_final_final_output'
        empty_dict = get_empty_dict(empty_path)

        npy_save_path = os.path.join(self.model_save_path, 'fold_' + str(fold_index), self.pretrained_model)
        if not os.path.exists(npy_save_path):
            os.makedirs(npy_save_path)

        self.load_pretrained_model(fold_index)
        self.G.eval()

        test_dir = r'/data/shentao/Airbus/AirbusShipDetectionChallenge/test_v2'
        # test_loader = get_5foldloader(self.image_size, 1, 0, mode='test')

        all_test = os.listdir(test_dir)
        nonempty_test = []
        for img in all_test:
            if img+'.npy' not in empty_dict:
                nonempty_test.append(img)

        print(len(nonempty_test))
        random.shuffle(nonempty_test)

        c = 0
        for id in nonempty_test:
            image_path = os.path.join(test_dir, id)
            save_npy = os.path.join(npy_save_path, id+'.png')

            if (not os.path.exists(save_npy)):
                output_mat = self.infer_one_img_from_path_8(image_path)
                output_mat = output_mat.reshape([self.image_size, self.image_size])

                output_mat = output_mat*255
                output_mat[output_mat > 255] = 255
                output_mat[output_mat < 0] = 0

                output_mat = output_mat.astype(np.uint8)
                # np.save(save_npy, output_mat)
                cv2.imwrite(save_npy, output_mat)

            if c % 500 == 0 and c > 0:
                print(c)
            c += 1

        # submission = create_submission(output_list)
        # submission.to_csv(self.model_name+'_'+str(self.pretrained_model)+
        #                   '_'+str(fold_index)+'.csv', index=None)

def main(config):
    # For fast training
    cudnn.benchmark = True
    if config.mode == 'train':
        solver = DecoderSolver(config)
        solver.train_fold(config.train_fold_index)
    if config.mode == 'val_fold':
        solver = DecoderSolver(config)
        solver.val(config.train_fold_index, True)
    if config.mode == 'val_fold_NoTTA':
        solver = DecoderSolver(config)
        val_loader = get_5foldloader(config.image_size, config.batch_size, config.train_fold_index, mode = 'val')
        solver.val_NoTTA(config.train_fold_index, True, val_loader)
    if config.mode == 'test_fold':
        solver = DecoderSolver(config)
        solver.infer(config.train_fold_index)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    parser.add_argument('--train_fold_index', type=int, default = 0)

    parser.add_argument('--model', type=str, default='model34_DeepSupervised_Dblock')
    parser.add_argument('--fmap', type=int, default=32)
    parser.add_argument('--model_name', type=str, default='model34_DeepSupervised_Dblock_resize768')
    parser.add_argument('--image_size', type=int, default=768)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--mode', type=str, default='test_fold', choices=['train', 'val','val_fold','val_fold_NoTTA','test','test_fold'])

    parser.add_argument('--pretrained_model', type=str, default='12_final')

    parser.add_argument('--is_freeze', type=bool, default=False)
    parser.add_argument('--is_bottleneck', type=bool, default=True)

    parser.add_argument('--dice_weight', type=float, default=0.5)
    parser.add_argument('--bce_weight', type=float, default=0.9)

    # Model hyper-parameterse;/
    parser.add_argument('--num_epochs', type=int, default=80)
    parser.add_argument('--lr_steps', type=str, default='20,30')
    # parser.add_argument('--lr_steps', type=str, default='10,20')
    parser.add_argument('--lr', type=float, default=0.0001)

    parser.add_argument('--num_iters', type=int, default=200000)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)

    # Test settings
    parser.add_argument('--test_model', type=str, default='10000')
    parser.add_argument('--image_path', type=str, default=r'')
    parser.add_argument('--rafd_image_path', type=str, default=r'')

    parser.add_argument('--log_path', type=str, default='logs')
    parser.add_argument('--model_save_path', type=str, default='models')
    parser.add_argument('--sample_path', type=str, default='samples')
    parser.add_argument('--result_path', type=str, default='results')

    # Step size
    parser.add_argument('--log_step', type=int, default=100)
    parser.add_argument('--sample_step', type=int, default=2)
    parser.add_argument('--model_save_step', type=int, default=20000)

    config = parser.parse_args()
    print(config)
    main(config)