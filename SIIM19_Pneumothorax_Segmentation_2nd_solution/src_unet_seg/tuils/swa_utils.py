from tqdm import tqdm

import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.zeros_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    model.cuda()
    torch.nn.DataParallel(model)
    for input_dict in tqdm(loader):
        inputs, _ = input_dict
        inputs = inputs.cuda(async=True)
        input_var = torch.autograd.Variable(inputs)
        b = input_var.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input_var)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))
# def bn_update(loader, model, device):
#     """
#         BatchNorm buffers update (if any).
#         Performs 1 epochs to estimate buffers average using train dataset.
#         :param loader: train dataset loader for buffers average estimation.
#         :param model: model being update
#         :return: None
#     """
#     if not check_bn(model):
#         return
#     model.train()
#     momenta = {}
#     model.apply(reset_bn)
#     model.apply(lambda module: _get_momenta(module, momenta))
#     n = 0
#
#     model = model.to(device)
#     pbar = tqdm(loader, unit="samples", unit_scale=loader.batch_size)
#     for batch in pbar:
#         # images = batch.get('input', None)
#         # if images is None:
#         #     warnings.warn("empty inputs")
#         #     continue
#
#         images = images.to(device)
#         b = images.size(0)
#
#         momentum = b / (n + b)
#         for module in momenta.keys():
#             module.momentum = momentum
#
#         model(images)
#         n += b
#
#     model.apply(lambda module: _set_momenta(module, momenta))


def detach_params(model):
    for param in model.parameters():
        param.detach_()

    return model


def evaluate(loader, model):
    model.eval()

    out_pred = torch.FloatTensor().cuda()
    out_gt = torch.FloatTensor().cuda()

    for input_dict in loader:
        inputs, targets = input_dict
        inputs = inputs.cuda()
        targets = targets.cuda().float()

        logits = model(inputs)
        probabilities = torch.sigmoid(logits)

        out_pred = torch.cat((out_pred, probabilities), 0)
        out_gt = torch.cat((out_gt, targets), 0)

    eval_metric_bundle = search_f2(out_pred, out_gt)
    print('===> Best', eval_metric_bundle)
    
    return eval_metric_bundle


def choose_device(device):
    if not isinstance(device, str):
        return device

    if device not in ['cuda', 'cpu']:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda":
        assert torch.cuda.is_available()

    device = torch.device(device)
    return device


def swa(load_model, model, model_folder, dataset, batch_size, device, e0, e1, e2, num_fold):
    directory = Path(model_folder)
    model_files = [f for f in directory.iterdir() if (str(f).endswith(str(e0) + '_' + str(num_fold) + ".pth.tar")
                                                      or str(f).endswith(str(e1) + '_' + str(num_fold) + ".pth.tar")
                                                      or str(f).endswith(str(e2) + '_' + str(num_fold) + ".pth.tar"))]
    assert(len(model_files) > 1)

    net = load_model(model_files[0], model)
    for i, f in enumerate(model_files[1:]):
        net2 = load_model(f, model)
        moving_average(net, net2, 1. / (i + 2))

    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, drop_last=True)
    # device = choose_device(device)
    with torch.no_grad():
        bn_update(dataloader, net)
    return net
