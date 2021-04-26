import torch.nn as nn
import argparse
import os

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import random

import numpy as np

from tqdm import tqdm
from tools.read_data_multi import DataSet
from tools.k_fold import k_fold_pre
from models import ResNet
from tools.Regularization import Regularization
from sklearn.metrics import roc_auc_score
from tools.classification import classification_LinearRegression


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='PyTorch ResNet Training')
parser.add_argument('--model', default='ResNet', type=str, help='baseline of the model')
parser.add_argument('--fold', default=5, type=int, help='number of k-fold')
parser.add_argument('--fold_index', default=4, type=int,help='index of k-fold(0-4)')
parser.add_argument('--epochs', default=20, type=int,help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,help='mini-batch size (default: 64)')
parser.add_argument('--num_classes', default=2, type=int, help='numbers of classes (default: 1)')
parser.add_argument('--mcbp', '--compact-bilinear-pooling', default='mbp', help='compact bilinear pooling(default:False)')
parser.add_argument('--optimizer', default='SGD', type=str, help='optimizer (SGD)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,help='print frequency (default: 10)')
parser.add_argument('--growth', default=32, type=int,help='number of new channels per layer (default: 12)')
parser.add_argument('--p', default=1, type=int, help='norm (default: 2)')
parser.add_argument('--weight-decay-fc', '--wdfc', default=0, type=float,help='weight decay fc (default: 1e-4)')
parser.add_argument('--seed', default=2, type=int, help='random seed(default: 1)')
parser.add_argument('--times', default=[27, 27], nargs='+', type=int, help='times_1,times_0')
parser.add_argument('--dx', default=0, type=int,  help=' ')  # 偏移像素百分比 1/30 * dx
parser.add_argument('--dy', default=0, type=int,  help=' ')
parser.add_argument('--dz', default=0, type=int,  help=' ')
parser.add_argument('--resume',
                    default='./result/file_name/checkpoint/',
                    type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='file_name',
                    type=str, help='name of experiment')
parser.add_argument('--use_cuda', default=True, help='whether to use_cuda(default: True)')
args = parser.parse_args()
DATA_DIR = './data/'
DATA_IMAGE_LIST = './label/TP53_3D_ADC_DWI_T2.txt'
MODEL_DIR = '/result/'

F = nn.Softmax(dim=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not os.path.exists(MODEL_DIR+args.name):
    os.makedirs(MODEL_DIR+args.name)


def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    global best_prec1, use_cuda
    best_prec1 = 0

    seed_torch(seed=args.seed)
    use_cuda = True
    use_cuda = use_cuda and torch.cuda.is_available()

    # create model
    model = ResNet.Multimodal_ResNet(num_class=args.num_classes, mcbp=args.mcbp, pretrained=True)

    if use_cuda:
        model = model.cuda()
        # for training on multiple GPUs.
        # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
        # model = torch.nn.DataParallel(model).cuda()
    # cudnn.benchmark = True
    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # define loss function (criterion) and optimizer
    # criterion = FocalLoss(class_num=args.num_classes, alpha=None, gamma=2, size_average=True)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    if args.weight_decay_fc > 0:
        reg_loss = Regularization(args.weight_decay_fc, p=args.p).to(device)
    else:
        reg_loss = 0
        print("no regularization")
    # compute by compute_mean_std
    # normalize = transforms.Normalize([0.0573, 0.0573, 0.0573], [0.1102, 0.1102, 0.1102])  # nrrd
    train_names, val_names = k_fold_pre(MODEL_DIR+"data_fold.txt", image_list_file=DATA_IMAGE_LIST, fold=args.fold)

    k =args.fold_index
    if args.resume:
        if os.path.isfile(args.resume + 'checkpoint' + str(k) + '.pth.tar'):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume + 'checkpoint' + str(k) + '.pth.tar')
            args.start_epoch = checkpoint['epoch']
            best_prec = checkpoint['best_prec1']
            print('best_prec1:', best_prec)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            print("=> use initial checkpoint")
            checkpoint = torch.load(MODEL_DIR + "%s/checkpoint_init.pth.tar" % args.name)
            model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        return 0
    result_filename = MODEL_DIR + "{}/test_result.txt".format(args.name, str(k))
    result_file = open(result_filename, 'a')  # 'a'  'w+' 追加
    for v in range(2):
        if v == 0:
            filename = MODEL_DIR + "{}/fold_{}_result_train.txt".format(args.name, str(k))
            if os.path.exists(filename):
                continue
            val_dataset = DataSet(data_dir=DATA_DIR, image_list_file=DATA_IMAGE_LIST, fold=train_names[k],
                                  transform=True, fold_num=k, filepath=args, mode='train')
        else:
            filename = MODEL_DIR + "{}/fold_{}_result_test.txt".format(args.name, str(k))
            if os.path.exists(filename):
                continue
            val_dataset = DataSet(data_dir=DATA_DIR, image_list_file=DATA_IMAGE_LIST, fold=val_names[k],
                                  transform=True, fold_num=k, filepath=args, mode='test')  #

        kwargs = {'num_workers': 8, 'pin_memory': True}
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch_size, **kwargs)
        epoch = args.start_epoch - 1
        # evaluate on validation set
        val_losses, val_acc, val_auc, output_val, label_val = validate(val_loader, model, criterion, reg_loss, epoch, k)
        file = open(filename, 'w')  # 'a'  'w+' 追加
        F_s = nn.Softmax(dim=0)
        for i in range(output_val.size()[0]):
            output = F_s(output_val[i])
            out_write = str(output.cpu().numpy()[1]) + ' ' + str(int(label_val[i][1])) + '\n'
            out_write = out_write.replace('[', '').replace(']', '')
            file.write(out_write)
        file.close()

        result_file.write(str(k) + ' ' + str(val_acc.avg) + ' ' + str(val_auc) + '\n')
        result_file.close()

        classification_LinearRegression(path=MODEL_DIR + "%s/" % args.name,
                                        train_patient_file="fold_{}_image_names_train.txt".format(k),
                                        train_slice_result_file="fold_{}_result_train.txt".format(k),
                                        test_patient_file="fold_{}_image_names_test.txt".format(k),
                                        test_slice_result_file="fold_{}_result_test.txt".format(k),
                                        fold=k, times_1=args.times[0], times_0=args.times[1])
        print('Tests have finished')


def validate(val_loader, model, criterion, reg_loss, epoch, fold):  # 返回值为准确率
    """Perform validation on the validation set"""
    val_losses = AverageMeter()
    val_acc = AverageMeter()

    # switch to evaluate mode  切换到评估模式
    model.eval()  # 很重要   how to move
    # model.train()
    target_roc = torch.zeros((0, args.num_classes))
    output_roc = torch.zeros((0, args.num_classes))

    with torch.no_grad():
        with tqdm(val_loader, ncols=130) as t:
            for i, (input1, input2, input3, target, _) in enumerate(t):
                t.set_description("valid epoch %s" % epoch)
                if use_cuda:
                    target = target.type(torch.LongTensor).cuda()
                    input1 = input1.type(torch.FloatTensor).cuda()
                    input2 = input2.type(torch.FloatTensor).cuda()
                    input3 = input3.type(torch.FloatTensor).cuda()
                else:
                    target = target.type(torch.LongTensor)
                    input1 = input1.type(torch.FloatTensor)
                    input2 = input2.type(torch.FloatTensor)
                    input3 = input3.type(torch.FloatTensor)
                # compute output
                output, output1, output2, output3 = model(input1, input2, input3)  #
                # output = model(input1, input2, input3)

                if len(output.shape) == 1:
                    output = output.unsqueeze(0)
                val_loss = criterion(output, target.squeeze(1))
                if args.weight_decay_fc > 0:
                    val_loss = val_loss + reg_loss(model.fc1) + reg_loss(model.fc2) + reg_loss(model.fc3)
                val_losses.update(val_loss.item(), input1.size(0))

                target_roc = torch.cat((target_roc, torch.zeros(target.shape[0], args.num_classes).scatter_(1, target.cpu(), 1)), dim=0)
                output_roc = torch.cat((output_roc, output.data.cpu()), dim=0)
                # -------------------------------------Accuracy--------------------------------- #
                acc = accuracy(output.data, target, input1)  # 一个batchsize中n类的平均准确率  输出为numpy类型
                val_acc.update(acc, input1.size(0))

                t.set_postfix({
                    'loss': '{loss.val:.4f}({loss.avg:.4f})'.format(loss=val_losses),
                    'Acc': '{acc.val:.4f}({acc.avg:.4f})'.format(acc=val_acc)}
                )

    # -------------------------------------AUROC------------------------------------ #
    AUROC = aucrocs(output_roc, target_roc)
    print('The AUROC is %.4f' % AUROC)
    # -------------------------------------AUROC------------------------------------ #

    return val_losses, val_acc, AUROC, output_roc, target_roc


def accuracy(output, target, input):
    target = torch.zeros(target.shape[0], args.num_classes).scatter_(1, target.cpu(), 1)  # 转化为为 one-hot ([32, 2])
    output = F(output)
    output_np = output.cpu().numpy()
    target_np = target.cpu().numpy()

    output_arg = np.argsort(output_np, axis=1)  # 返回索引，判断0和1哪个大
    target_arg = np.argsort(target_np, axis=1)
    error = target_arg[:, 1] ^ output_arg[:, 1]
    error_rate = error.sum() / input.shape[0]
    acc = 1 - error_rate
    return acc


def aucrocs(output, target):  # 改准确度的计算方式

    """
    Returns:
    List of AUROCs of all classes.
    """
    output = F(output)
    output_np = output.cpu().numpy()
    # print('output_np:',output_np)
    target_np = target.cpu().numpy()
    # print('target_np:',target_np)
    AUROCs = roc_auc_score(target_np[:, 1], output_np[:, 1])

    return AUROCs


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
