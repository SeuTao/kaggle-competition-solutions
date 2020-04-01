from net.imagenet_pretrain_model.xception import *
from net.archead import *
BatchNorm2d = nn.BatchNorm2d

###########################################################################################3
class Net(nn.Module):

    def load_pretrain(self, pretrain_file, is_skip_fc = False):
        pretrain_state_dict = torch.load(pretrain_file)
        state_dict = self.state_dict()
        keys = list(state_dict.keys())
        for key in keys:
            if is_skip_fc and 'fc' in key:
                print('skip: ' + key)
                continue
            state_dict[key] = pretrain_state_dict[r'module.'+key]

        self.load_state_dict(state_dict)
        print('')

    def __init__(self, num_class=340, is_arc = False, arc_s = None, arc_m = None):
        super(Net,self).__init__()
        self.basemodel = xception()
        self.basemodel.conv1 = nn.Conv2d(6, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.basemodel.last_linear = nn.Sequential()

        self.is_arc = is_arc
        if self.is_arc:
            self.fc = ArcModule(2048, num_class, s=arc_s, m=arc_m)
        else:
            self.fc = nn.Sequential(nn.Dropout(),
                                    nn.Linear(2048, num_class))


    def forward(self, x, label = None):
        x = self.basemodel.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        fea = x.view(x.size(0), -1)

        if self.is_arc:
            x = F.normalize(fea)
            x = self.fc(x, label)
        else:
            x = self.fc(fea)
        return x, fea

    def set_mode(self, mode, is_freeze_bn=False ):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
            if is_freeze_bn==True: ##freeze
                for m in self.modules():
                    if isinstance(m, BatchNorm2d):
                        m.eval()
                        m.weight.requires_grad = False
                        m.bias.requires_grad   = False

### run ##############################################################################
def run_check_net():
    net = Net(num_class=1000).cuda()
    net.set_mode('train')
    print(net)


########################################################################################
if __name__ == '__main__':
    # print( '%s: calling main function ... ' % os.path.basename(__file__))
    run_check_net()
    print( 'sucessful!')