import argparse
from data_loader import *
from torch.backends import cudnn
from torchvision.utils import save_image
from model.model_stnet import StNetV2, StNetV2_slim, StNetV2_slim_ChangePool,StNetV2_slim_ChangePool_Dblock
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
            self.G = StNetV2_slim(num_classes=1, num_filters=self.fmap, pretrained=True)

        elif self.model == 'stnet34_V2_slim_ChangPool':
            self.G = StNetV2_slim_ChangePool()

        elif self.model == 'stnet34_V2_slim_ChangPool_Dblock':
            self.G = StNetV2_slim_ChangePool_Dblock()


        self.g_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.G.parameters()),
                                           self.g_lr, weight_decay=0.0002, momentum=0.9)
        self.print_network(self.G, 'G')

        # self.print_network(self.C, 'C')
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

            self.G.load_state_dict(torch.load(os.path.join(self.model_save_path,'fold_' + str(fold_index),
                                                           '{}_G.pth'.format(self.pretrained_model))))
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
        if not os.path.exists(os.path.join(self.model_save_path,'fold_'+str(fold_index))):
            os.makedirs(os.path.join(self.model_save_path,'fold_'+str(fold_index)))

        data_loader = get_5foldloader(self.image_size, self.batch_size, fold_index, mode= 'train')
        val_loader = get_5foldloader(self.image_size, self.batch_size, fold_index, mode = 'val')

        iters_per_epoch = len(data_loader)
        for i, (images, labels, _) in enumerate(data_loader):
            fixed_images = self.to_var(images, volatile=True)
            fixed_labels = self.to_var(labels, volatile=True)
            break

        save_fix_image = self.denorm_image(fixed_images.data)
        save_fix_label = self.denorm_output(fixed_labels.data)

        # if self.pretrained_model:
        #     start = int(self.pretrained_model.split('_')[0])
        # else:
        start = 0

        lr_tmp = self.g_lr
        # Start training
        start_time = time.time()
        for e in range(start, self.num_epochs):
            for i, (images, labels, _) in enumerate(data_loader):
                inputs = self.to_var(images)
                labels = self.to_var(labels)

                # Compute loss with fake images
                output_logits, output = self.G(inputs)
                bce_loss = mixed_dice_bce_loss(output_logits, labels, dice_weight=self.dice_weight, bce_weight=self.bce_weight)

                self.g_optimizer.zero_grad()
                bce_loss.backward()
                self.g_optimizer.step()

                # # Logging
                loss = {}
                loss['D/loss_real'] = bce_loss.data[0]

                # Print out log info
                if (i+1) % self.log_step == 0:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))

                    log = self.model_name+" FOLD: {}, Elapsed [{}], Epoch [{}/{}], Iter [{}/{}], lr {}".format(
                        fold_index, elapsed, e+1, self.num_epochs, i+1, iters_per_epoch, lr_tmp)

                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

                if (i + 1) % 1000 == 0:
                    fixed_output = self.G(fixed_images)
                    save_fix_out = self.denorm_output(fixed_output[1].data)
                    save = torch.cat([save_fix_label, save_fix_out], dim=3)
                    save_image(save, os.path.join(self.sample_path, '{}_{}_output.png'.format(e + 1, i + 1)),
                               nrow=1, padding=0)
                    print('Translated images and saved into {}..!'.format(self.sample_path))

                    self.val_NoTTA(0, is_load=False, val_loader=val_loader)

            # Decay learning rate
            if (e+1) in self.lr_steps:
                lr_tmp *= 0.1
                self.update_lr(lr_tmp)
                print ('Decay learning rate to g_lr: {}'.format(lr_tmp))

            if (e+1) % self.sample_step == 0:
                torch.save(self.G.state_dict(),os.path.join(self.model_save_path,
                                                            'fold_'+str(fold_index),
                                                            '{}_final_G.pth'.format(e + 1)))

            self.val_NoTTA(0, is_load=False, val_loader=val_loader)

    def val(self, fold_index, is_load):
        if is_load:
            self.load_pretrained_model(fold_index)

        self.G.eval()

        val_dir = r'/data/shentao/Airbus/AirbusShipDetectionChallenge_384/train'
        val_loader = get_5foldloader(self.image_size, 1, fold_index, mode = 'val')

        iou_list = []
        for i, (image_path, images, labels) in enumerate(val_loader):
                labels = self.to_var(labels)
                label_mat = labels.data
                label_mat = label_mat.cpu().numpy()

                image_path = os.path.join(val_dir,image_path[0])
                output_mat = self.infer_one_img_from_path_8(image_path)

                label_mat[label_mat > 0.5] = 1
                label_mat[label_mat <= 0.5] = 0
                output_mat[output_mat > 0.5] = 1
                output_mat[output_mat <= 0.5] = 0

                label_mat = label_mat.astype(np.uint8)
                output_mat = output_mat.astype(np.uint8)

                if np.sum(label_mat) > 0 or np.sum(output_mat) > 0:
                    iou_tmp_1 = iou(label_mat, output_mat)
                    iou_tmp = iou_tmp_1
                    iou_list.append(iou_tmp)

        print(sum(iou_list)/len(iou_list))
        self.G.train()

        return sum(iou_list)/len(iou_list)

    def val_NoTTA(self, fold_index, is_load, val_loader):

        if is_load:
            self.load_pretrained_model(fold_index)
        self.G.eval()

        iou_list = []
        for i, ( _, images, labels, _) in enumerate(val_loader):
                labels = self.to_var(labels)
                images = self.to_var(images)

                label_mat = labels.data
                label_mat = label_mat.cpu().numpy()

                output_mat = self.G.forward(images)[1].data.cpu().numpy()

                label_mat = label_mat.reshape([-1,768,768])
                output_mat = output_mat.reshape([-1,self.image_size,self.image_size])

                # print(label_mat.shape)
                # print(output_mat.shape)
                for sample_i in range(label_mat.shape[0]):
                    label_tmp = label_mat[sample_i]
                    output_tmp = output_mat[sample_i]

                    label_tmp[label_tmp > 0.5] = 1
                    label_tmp[label_tmp <= 0.5] = 0
                    output_tmp[output_tmp > 0.5] = 1
                    output_tmp[output_tmp <= 0.5] = 0

                    label_tmp = label_tmp.astype(np.uint8)
                    output_tmp = output_tmp.astype(np.uint8)
                    output_tmp = cv2.resize(output_tmp,(768,768))
                    iou_list.append(f2([label_tmp],[output_tmp]))

        Valid_F2 =  sum(iou_list)/(1.0*len(iou_list))
        print('Valid F2: %.3f'%(Valid_F2))
        self.G.train()
        return sum(iou_list)/len(iou_list)

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
        img_all = self.G.forward(img_all)[1].data.cpu().numpy()

        mask0 = img_all[0:4]
        mask1 = img_all[4:8].transpose(0,1,3,2)

        mask = mask0 + mask1
        mask = mask[0:1] + mask[1:2][:, :, ::-1, :] + mask[2:3][:, :, :, ::-1] + mask[3:4][:, :, ::-1, ::-1]
        return mask / 8.0

    def infer(self, fold_index):
        npy_save_path = os.path.join(self.model_save_path, 'fold_' + str(fold_index), self.pretrained_model)
        if not os.path.exists(npy_save_path):
            os.makedirs(npy_save_path)

        self.load_pretrained_model(fold_index)
        self.G.eval()

        test_dir = r'/data/shentao/Airbus/AirbusShipDetectionChallenge_384/test'
        test_loader = get_5foldloader(self.image_size, 1, 0, mode='test')

        # output_list = []
        for i, (id) in enumerate(test_loader):
            image_path = os.path.join(test_dir, id[0])
            output_mat = self.infer_one_img_from_path_8(image_path)
            output_mat = output_mat.reshape([self.image_size, self.image_size])

            output_mat[output_mat > 0.5] = 1
            output_mat[output_mat <= 0.5] = 0
            output_mat = output_mat.astype(np.uint8)

            np.save(os.path.join(npy_save_path, id[0]), output_mat)

            if i % 1000 == 0 and i > 0:
                print(i)

        # submission = create_submission(output_list)
        # submission.to_csv(self.model_name+'_'+str(self.pretrained_model)+
        #                   '_'+str(fold_index)+'.csv', index=None)

    def infer_5fold(self):
        self.G.eval()
        test_dir = r'/data/shentao/Airbus/AirbusShipDetectionChallenge_384/test'
        test_loader = get_5foldloader(self.image_size, 1, 0, mode='test')

        predict_dict = {}
        for fold_index in range(5):
            self.load_pretrained_model(fold_index)
            for i, (id) in enumerate(test_loader):
                    image_path = os.path.join(test_dir, id[0])
                    output_mat = self.infer_one_img_from_path_8(image_path)
                    output_mat = output_mat.reshape([self.image_size, self.image_size])

                    output_mat[output_mat > 1.0] = 1.0
                    output_mat[output_mat < 0.0] = 0.0

                    if id[0] not in predict_dict:
                        predict_dict[id[0]] = output_mat
                    else:
                        predict_dict[id[0]] += output_mat

                    if i%1000 == 0 and i>0:
                        print(self.model_name + ' fold index: '+str(fold_index) +' '+str(i))

        out = []
        for id in predict_dict:
            output_mat = predict_dict[id] / 5.0
            output_mat[output_mat > 0.5] = 1
            output_mat[output_mat <= 0.5] = 0
            output_mat = output_mat.astype(np.uint8)
            out.append([id, output_mat])

        submission = create_submission(out, 768,768)
        submission.to_csv(self.model_name + '_'+str(self.pretrained_model)+'_5fold.csv', index=None)

    def val_5fold(self):
        fold_0 = self.val(0, True)
        fold_1 = self.val(1, True)
        fold_2 = self.val(2, True)
        fold_3 = self.val(3, True)
        fold_4 = self.val(4, True)

        ave = fold_0 + fold_1 + fold_2 + fold_3 + fold_4
        ave /= 5.0
        print(self.pretrained_model)
        print('Five Fold: '+'%.3f|%.3f|%.3f|%.3f|%.3f|%.3f|' % (fold_0,fold_1,fold_2,fold_3,fold_4,ave))


def main(config):
    # For fast training
    cudnn.benchmark = True
    if config.mode == 'train':
        solver = DecoderSolver(config)
        solver.train_fold(config.train_fold_index)
    if config.mode == 'val':
        solver = DecoderSolver(config)
        solver.val_5fold()
    if config.mode == 'val_fold':
        solver = DecoderSolver(config)
        solver.val(config.train_fold_index, True)
    if config.mode == 'val_fold_NoTTA':
        solver = DecoderSolver(config)
        val_loader = get_5foldloader(config.image_size, config.batch_size, 0, mode = 'val')

        solver.pretrained_model = '40_final'
        solver.val_NoTTA(config.train_fold_index, True, val_loader)
        solver.pretrained_model = '45_final'
        solver.val_NoTTA(config.train_fold_index, True, val_loader)
        solver.pretrained_model = '50_final'
        solver.val_NoTTA(config.train_fold_index, True, val_loader)
        solver.pretrained_model = '55_final'
        solver.val_NoTTA(config.train_fold_index, True, val_loader)
        solver.pretrained_model = '60_final'
        solver.val_NoTTA(config.train_fold_index, True, val_loader)
        solver.pretrained_model = '65_final'
        solver.val_NoTTA(config.train_fold_index, True, val_loader)
        solver.pretrained_model = '70_final'
        solver.val_NoTTA(config.train_fold_index, True, val_loader)

    if config.mode == 'test':
        solver = DecoderSolver(config)
        solver.infer_5fold()
    if config.mode == 'test_fold':
        solver = DecoderSolver(config)
        solver.infer(config.train_fold_index)
        # solver.infer()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    parser.add_argument('--train_fold_index', type=int, default = 0)

    parser.add_argument('--model', type=str, default='stnet34_V2_slim_ChangPool_Dblock')
    parser.add_argument('--fmap', type=int, default=32)
    parser.add_argument('--model_name', type=str, default='stnet34_V2_slim_ChangPool_Dblock_resize576')
    parser.add_argument('--image_size', type=int, default=576)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--mode', type=str, default='val_fold_NoTTA', choices=['train', 'val','val_fold','val_fold_NoTTA','test','test_fold'])

    parser.add_argument('--pretrained_model', type=str, default='6_final')

    parser.add_argument('--is_freeze', type=bool, default=False)
    parser.add_argument('--is_bottleneck', type=bool, default=True)

    parser.add_argument('--dice_weight', type=float, default=0.5)
    parser.add_argument('--bce_weight', type=float, default=0.9)

    # Model hyper-parameters
    parser.add_argument('--num_epochs', type=int, default=80)
    parser.add_argument('--lr_steps', type=str, default='5,10,20')
    parser.add_argument('--lr', type=float, default=0.001)

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