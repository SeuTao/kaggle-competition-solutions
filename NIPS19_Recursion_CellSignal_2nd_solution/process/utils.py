from include import *
from torch.autograd import Variable

def save(list_or_dict,name):
    f = open(name, 'w')
    f.write(str(list_or_dict))
    f.close()

def load(name):
    f = open(name, 'r')
    a = f.read()
    tmp = eval(a)
    f.close()
    return tmp

def dot_numpy(vector1 , vector2,emb_size = 512):
    vector1 = vector1.reshape([-1, emb_size])
    vector2 = vector2.reshape([-1, emb_size])
    vector2 = vector2.transpose(1,0)

    cosV12 = np.dot(vector1, vector2)
    return cosV12

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def softmax_cross_entropy_criterion(logit, truth, is_average=True):
    loss = F.cross_entropy(logit, truth, reduce=is_average)
    return loss

def softmax_add_newwhale(logit, truth):
    indexs_NoNew = (truth != 5004).nonzero().view(-1)
    indexs_New = (truth == 5004).nonzero().view(-1)

    logits_NoNew = logit[indexs_NoNew]
    truth_NoNew = truth[indexs_NoNew]

    logits_New = logit[indexs_New]
    print(logits_New.size())

    if logits_NoNew.size()[0]>0:
        loss = nn.CrossEntropyLoss(reduce=True)(logits_NoNew, truth_NoNew)
    else:
        loss = 0

    if logits_New.size()[0]>0:
        logits_New = torch.softmax(logits_New,1)
        logits_New = logits_New.topk(1,1,True,True)[0]
        target_New = torch.zeros_like(logits_New).float().cuda()
        loss += nn.L1Loss()(logits_New, target_New)

    return loss

def metric(logit, truth, is_average=True, is_prob = False):
    if is_prob:
        prob = logit
    else:
        prob = F.softmax(logit, 1)

    value, top = prob.topk(5, dim=1, largest=True, sorted=True)
    correct = top.eq(truth.view(-1, 1).expand_as(top))

    if is_average==True:
        # top-3 accuracy
        correct = correct.float().sum(0, keepdim=False)
        correct = correct/len(truth)

        top = [correct[0],
               correct[0] + correct[1],
               correct[0] + correct[1] + correct[2],
               correct[0] + correct[1] + correct[2] + correct[3],
               correct[0] + correct[1] + correct[2] + correct[3] + correct[4]]

        precision = correct[0] / 1 + correct[1] / 2 + correct[2] / 3 + correct[3] / 4 + correct[4] / 5

        return precision, top
    else:
        return correct

def metric_top1(logit, truth, is_average=True, is_prob = False):
    if is_prob:
        prob = logit
    else:
        prob = F.softmax(logit, 1)

    value, top = prob.topk(1, dim=1, largest=True, sorted=True)
    correct = top.eq(truth.view(-1, 1).expand_as(top))

    if is_average==True:
        # top-3 accuracy
        correct = correct.float().sum(0, keepdim=False)
        correct = correct/len(truth)
        top = [correct[0]]
        precision = correct[0]

        return precision, top
    else:
        return correct


def top_n_np(preds, labels):
    n = 5
    predicted = np.fliplr(preds.argsort(axis=1)[:, -n:])
    top5 = []

    re = 0
    for i in range(len(preds)):
        predicted_tmp = predicted[i]
        labels_tmp = labels[i]
        for n_ in range(5):
            re += np.sum(labels_tmp == predicted_tmp[n_]) / (n_ + 1.0)

    re = re / len(preds)

    for i in range(n):
        top5.append(np.sum(labels == predicted[:, i])/ (1.0*len(labels)))

    return re, top5

def metric_binary(logit, truth):
    prob = F.softmax(logit, 1)
    value, top = prob.topk(2, dim=1, largest=True, sorted=True)
    correct = top.eq(truth.view(-1, 1).expand_as(top))
    correct = correct.float().sum(0, keepdim=False)
    correct = correct / len(truth)
    return correct[0]

def metric_bce(logit, truth):
    prob = F.sigmoid(logit)
    prob[prob > 0.5] = 1
    prob[prob < 0.5] = 0
    correct =   prob.eq(truth.view(-1, 1).expand_as(prob))
    correct = correct.float().sum(0, keepdim=False)
    correct = correct/len(truth)
    return correct
