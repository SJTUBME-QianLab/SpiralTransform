import torch.nn as nn
import argparse
import os
import random
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from tools.Regularization import Regularization
import numpy as np

from tools.read_data_multi import DataSet_Mini
from tools.k_fold import k_fold_pre
from models import ResNet
from sklearn.metrics import roc_auc_score

import math
from tqdm import tqdm

# used for logging to TensorBoard
# from tensorboard_logger import log_value
from tensorboardX import SummaryWriter

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# CUDA_VISIBLE_DEVICES=1 python train.py
parser = argparse.ArgumentParser(description='PyTorch ResNet Training')
parser.add_argument('--model', default='ResNet', type=str, help='baseline of the model')
parser.add_argument('--pretrained', default=True, help='load pretrained model')
parser.add_argument('--fold_index', default=0, type=int, help='index of k-fold(0-4)')
parser.add_argument('--fold', default=5, type=int, help='number of k-fold')
parser.add_argument('--n_epoch', default=5, type=int, help='number of epoch to change')
parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int, help='mini-batch size (default: 64)')
parser.add_argument('--num_classes', default=2, type=int, help='numbers of classes (default: 2)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, help='initial learning rate')
parser.add_argument('--mcbp', '--compact-bilinear-pooling', default='mbp', help='compact bilinear pooling(default:False)')
parser.add_argument('--optimizer', default='SGD', type=str, help='optimizer (SGD)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,help='print frequency (default: 10)')
parser.add_argument('--growth', default=32, type=int, help='number of new channels per layer (default: 12)')
parser.add_argument('--p', default=1, type=int, help='norm (default: 2)')
parser.add_argument('--weight-decay-fc', '--wdfc', default=0.001, type=float, help='weight decay fc (default: 1e-4)')
parser.add_argument('--loss-type', default='regularizationL1_yL2', type=str,
                    help='(default: regularization,regularizationL1_yL2)')
parser.add_argument('--alpha', default=0.01, type=float, help='weight of yL2(default: 1)')
parser.add_argument('--gama', default=1, type=float, help='weight of crossentroy(default: 1)')
parser.add_argument('--seed', default=2, type=int, help='random seed(default: 1)')
parser.add_argument('--num_workers', default=0, type=int, help='num_workers(default: 0)')
parser.add_argument('--dx', default=0, type=int,  help=' ')  # 偏移像素百分比 1/30 * dx
parser.add_argument('--dy', default=0, type=int,  help=' ')
parser.add_argument('--dz', default=0, type=int,  help=' ')
parser.add_argument('--resume',
                    default='./result/file_name/checkpoint/',
                    type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='file_name',
                    type=str, help='name of experiment')
parser.add_argument('--tensorboard', default=True,
                    help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--use_cuda', default=True, help='whether to use_cuda(default: True)')
args = parser.parse_args()  #

DATA_DIR = './data/'
DATA_IMAGE_LIST = './label/TP53_3D_ADC_DWI_T2.txt'
MODEL_DIR = '/result/'
F = nn.Softmax(dim=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    global best_prec_all, use_cuda, writer

    if args.tensorboard:
        # configure(MODEL_DIR + "%s" % args.name)
        writer = SummaryWriter(MODEL_DIR + "%s" % args.name)
    use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.seed > 0:
        seed_torch(args.seed)  # 固定随机数种子
    # create model
    model = ResNet.Multimodal_ResNet(num_class=args.num_classes, mcbp=args.mcbp, pretrained=args.pretrained)

    # input_random = torch.rand(32, 3, 100, 100)
    # if args.tensorboard:
    #     writer.add_graph(model, (input_random, input_random, input_random), True)
    if os.path.exists(MODEL_DIR + "%s/checkpoint_init.pth.tar" % args.name):
        checkpoint = torch.load(MODEL_DIR + "%s/checkpoint_init.pth.tar" % args.name)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        torch.save({'state_dict': model.state_dict()}, MODEL_DIR + "%s/checkpoint_init.pth.tar" % args.name)

    if use_cuda:
        model = model.cuda()
        # for training on multiple GPUs.
        # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
        # model = torch.nn.DataParallel(model).cuda()
    # get the number of model parameters
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # define optimizer
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    nesterov=True, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.99))
    else:
        print('Please choose true optimizer.')
        return 0

    # 5-fold 数据准备
    train_names, val_names = k_fold_pre(MODEL_DIR+"data_fold.txt", image_list_file=DATA_IMAGE_LIST,
                                        fold=args.fold)
    output, label, best_acc = [], [], []
    best_prec_all = 0  # 所有fold的概率
    fileaccauc_name = MODEL_DIR + "{}/fold_acc_auc.txt".format(args.name)
    fileaccauc = open(fileaccauc_name, 'a')
    for k in range(args.fold_index, args.fold_index+1):  # args.fold
        best_prec = 0  # 第k个fold的准确率
        # 读取第k个fold的数据
        train_dataset = DataSet_Mini(data_dir=DATA_DIR, image_list_file=DATA_IMAGE_LIST, fold=train_names[k],
                                     transform=True)  # normalize
        val_dataset = DataSet_Mini(data_dir=DATA_DIR, image_list_file=DATA_IMAGE_LIST, fold=val_names[k],
                                   transform=True)  #
        kwargs = {'num_workers': args.num_workers, 'pin_memory': True}
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)  # drop_last=True,
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

        # optionally resume from a checkpoint
        if args.resume:
            if os.path.isfile(args.resume + 'checkpoint' + str(k) + '.pth.tar'):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume + 'checkpoint' + str(k) + '.pth.tar')
                checkpoint_initial = torch.load(MODEL_DIR + "/%s/checkpoint_init.pth.tar" % args.name)
                model.load_state_dict(checkpoint_initial['state_dict'])
                # pretrained transfer mbp
                args.start_epoch = checkpoint['epoch']
                best_prec = checkpoint['best_prec1']
                print('best_prec1:', best_prec)
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
                print("=> use initial checkpoint")
                checkpoint = torch.load(MODEL_DIR + "%s/checkpoint_init.pth.tar" % args.name)
                model.load_state_dict(checkpoint['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return 0
        # define loss function
        criterion = loss_function(weight_decay_fc=args.weight_decay_fc, p=args.p)
        epoch_is_best = 0
        filename = MODEL_DIR + "{}/fold_avg_auc.txt".format(args.name)
        file = open(filename, 'a')  # 'a' 新建  'w+' 追加

        for epoch in range(args.start_epoch, args.epochs):

            adjust_learning_rate(optimizer, epoch)

            # train for one epoch
            train_losses, train_acc = train(train_loader, model, criterion, optimizer, epoch, k)

            # for name, layer in model.named_parameters():
            #     writer.add_histogram('fold' + str(k) + '/' + name + '_grad', layer.grad.cpu().data.numpy(), epoch)
            #     writer.add_histogram('fold' + str(k) + '/' + name + '_data', layer.cpu().data.numpy(), epoch)
            # evaluate on validation set
            val_losses, val_acc, prec1, output_val, label_val, AUROC = validate(val_loader, model, criterion, epoch, k)
            print('Accuracy {val_acc.avg:.4f}\t AUC {auc:.4f}'.
                  format(val_acc=val_acc, auc=AUROC))
            # 验证集用于验证训练的结果，因此就不用庞大的训练集来验证了，每迭代依次就要进行一次验证
            if args.tensorboard:
                # x = model.conv1.weight.data
                # x = vutils.make_grid(x, normalize=True, scale_each=True)
                # writer.add_image('data' + str(k) + '/weight0', x, epoch)  # Tensor
                writer.add_scalars('data' + str(k) + '/loss',
                                   {'train_loss': train_losses.avg, 'val_loss': val_losses.avg}, epoch)
                writer.add_scalars('data' + str(k) + '/Accuracy', {'train_acc': train_acc.avg, 'val_acc': val_acc.avg},
                                   epoch)
            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec
            if is_best == 1:
                epoch_is_best = epoch
                best_prec = max(prec1, best_prec)  # 这个fold的最高准确率
            best_prec_all = max(prec1, best_prec_all)  # 所有的最高准确率
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec,
            }, is_best, epoch, k)
            # 综合五次的output和label
            best_acc.append([best_prec, epoch_is_best])
            output.append(output_val)
            label.append(label_val)

            out_write = str(AUROC) + '\t'
            file.write(out_write)
        file.write('\n')
        file.close()

        acc_auc_out_write = str(train_acc.avg) + ' ' + str(val_acc.avg) + ' ' + str(AUROC) + '\n'
        fileaccauc.write(acc_auc_out_write)
        writer.close()
        print('fold_num: [{}]\t Best accuracy {} \t epoch {}'.format(k, best_prec, epoch_is_best))
    print('Best accuracy of all fold: ', best_prec_all)
    fileaccauc.close()
    state = {'output': output,
             'label': label,
             'best_acc': best_acc}
    torch.save(state, MODEL_DIR + "%s/output_label.pth.tar" % (args.name))


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
    # lr = args.lr * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
    # epoch = epoch - 150
    if epoch <= args.n_epoch:
        # lr = args.lr * epoch / args.n_epoch
        lr = args.lr
    else:
        lr = args.lr * (1 + np.cos((epoch - args.n_epoch) * math.pi / args.epochs)) / 2

    # log to TensorBoard
    if args.tensorboard:
        writer.add_scalar('learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(train_loader, model, criterion, optimizer, epoch, fold):
    """Train for one epoch on the training set"""
    train_losses = AverageMeter()
    train_acc = AverageMeter()
    # switch to train mode
    model.train()

    with tqdm(train_loader, ncols=130) as t:
        for i, (input1, input2, input3, target, _) in enumerate(t):
            t.set_description("train epoch %s" % epoch)
            if use_cuda:
                target = target.type(torch.LongTensor).cuda()
                input1 = input1.type(torch.FloatTensor).cuda()
                input2 = input2.type(torch.FloatTensor).cuda()
                input3 = input3.type(torch.FloatTensor).cuda()
            output, output1, output2, output3 = model(input1, input2, input3)
            # output = model(input1, input2, input3)
            # measure accuracy and record loss
            if len(output.shape) == 1:
                output = torch.unsqueeze(output, 0)
            if args.loss_type == 'regularization':
                train_loss = criterion.regularization(model, output, target.squeeze(1))
            elif args.loss_type == 'regularizationL1_yL2':
                train_loss = criterion.regularizationL1_yL2(model, output, output1[:, 1], output2[:, 1], output3[:, 1],
                                                            target.squeeze(1))
            else:
                print('Please input right loss-type.')
                return 0

            train_losses.update(train_loss.item(), input1.size(0))
            acc = accuracy(output.data, target, input1)
            train_acc.update(acc, input1.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            t.set_postfix({
                'loss': '{loss.val:.4f}({loss.avg:.4f})'.format(loss=train_losses),
                'Acc': '{acc.val:.4f}({acc.avg:.4f})'.format(acc=train_acc)}
            )

    # log to TensorBoard
    if args.tensorboard:
        writer.add_scalar('data' + str(fold) + '/train_loss', train_losses.avg, epoch)
        writer.add_scalar('data' + str(fold) + '/train_acc', train_acc.avg, epoch)
    return train_losses, train_acc


def validate(val_loader, model, criterion, epoch, fold):  # 返回值为准确率
    """Perform validation on the validation set"""
    # batch_time = AverageMeter()
    val_losses = AverageMeter()
    val_acc = AverageMeter()
    # switch to evaluate mode  切换到评估模式
    model.eval()  # 很重要
    target_roc = torch.zeros((0, args.num_classes))
    output_roc = torch.zeros((0, args.num_classes))
    # end = time.time()
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
                output, output1, output2, output3 = model(input1, input2, input3)
                # output = model(input1, input2, input3)

                # measure accuracy and record loss
                if len(output.shape) == 1:
                    output = torch.unsqueeze(output, 0)
                if args.loss_type == 'regularization':
                    val_loss = criterion.regularization(model, output, target.squeeze(1))
                elif args.loss_type == 'regularizationL1_yL2':
                    val_loss = criterion.regularizationL1_yL2(model, output, output1[:, 1], output2[:, 1], output3[:, 1],
                                                              target.squeeze(1))
                else:
                    print('Please input right loss-type.')
                    return 0
                val_losses.update(val_loss.item(), input1.size(0))

                target_roc = torch.cat(
                    (target_roc, torch.zeros(target.shape[0], args.num_classes).scatter_(1, target.cpu(), 1)), dim=0)
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
    # print('The AUROC is %.4f' % AUROC)
    # -------------------------------------AUROC------------------------------------ #

    # log to TensorBoard
    if args.tensorboard:
        writer.add_scalar('data' + str(fold) + '/val_loss', val_losses.avg, epoch)
        writer.add_scalar('data' + str(fold) + '/val_acc', val_acc.avg, epoch)
        writer.add_scalar('data' + str(fold) + '/val_AUC', AUROC, epoch)

    return val_losses, val_acc, val_acc.avg, output_roc, target_roc, AUROC


def accuracy(output, target, input):
    target = torch.zeros(target.shape[0], args.num_classes).scatter_(1, target.cpu(), 1)  # 转化为为 one-hot
    output = F(output)
    output_np = output.cpu().numpy()
    target_np = target.cpu().numpy()

    output_arg = np.argsort(output_np, axis=1)
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
    AUROCs = []
    for i in range(args.num_classes):
        AUROCs.append(roc_auc_score(target_np[:, i], output_np[:, i]))
    return AUROCs[args.num_classes - 1]


def save_checkpoint(state, is_best, epoch, fold):
    """Saves checkpoint to disk"""
    # filename = 'checkpoint' + str(fold) + '_' + str(epoch) + '.pth.tar'
    filename = 'checkpoint' + str(fold) + '.pth.tar'
    directory = MODEL_DIR + "%s/checkpoint/" % (args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        torch.save(state, filename)
        shutil.copyfile(filename, MODEL_DIR + '%s/checkpoint/' % (args.name) + 'model_best' + str(fold) + '.pth.tar')


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


def Model_single(model):
    model_s = []
    model_s_dict = []
    pretrained_dict = model.state_dict()
    for i in range(3):
        model_s.append(model.model1)
        model_s_dict.append(model_s[i].state_dict())
        # filename = MODEL_DIR + "{}/model_new_dict.txt".format(args.name)
        # file = open(filename, 'w')  # 'a' 新建  'w+' 追加
        for key, value in model_s_dict[i].items():
            if key == 'fc.0.weight':
                # print('{}<====={}'.format(key, key))
                w_length = pretrained_dict[key].shape[1] // 3
                model_s_dict[i][key] = pretrained_dict[key][:, w_length * i: w_length * (i + 1)]
            elif key == 'fc.0.bias':
                # print('{}<====={}'.format(key, key))
                model_s_dict[i][key] = pretrained_dict[key]
            else:
                # print('{}<====={}'.format(key, 'model' + str(i + 1) + key[5:]))
                model_s_dict[i][key] = pretrained_dict['model' + str(i + 1) + key[5:]]
            # out_write = key + '\n'
            # file.write(out_write)
        # file.close()
        model_s[i].load_state_dict(model_s_dict[i])
        if use_cuda:
            model_s[i] = model_s[i].cuda()

    return model_s


class loss_function:
    def __init__(self, weight_decay_fc=0, p=1):
        self.weight_decay = weight_decay_fc
        self.p = p
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def get_weight(self, model):
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization(self, model, output, target):
        if self.weight_decay == 0:
            loss = self.criterion(output, target)
        else:
            reg_loss = Regularization(self.weight_decay, p=self.p).to(device)
            loss = self.criterion(output, target) + reg_loss(model.fc1) + reg_loss(model.fc2) + reg_loss(model.fc3)
        return loss

    def regularizationL1_yL2(self, model, output, output1, output2, output3, target):
        loss_cro = self.criterion(output, target)
        batch, a = output.shape  # 32 2
        L = 0
        for s in range(batch):
            output_s = torch.zeros([3])
            output_s[0] = output1[s]
            output_s[1] = output2[s]
            output_s[2] = output3[s]
            for i in range(3):
                for j in range(i, 3):
                    L = L + torch.norm((output_s[i] - output_s[j]), p=2)
                    # L = L + torch.sqrt(((output_s[i] - output_s[j]) ** 2))
        if use_cuda:
            L = L.cuda()
        if self.weight_decay == 0:
            loss = loss_cro + args.alpha * L / batch
        else:
            reg_loss = Regularization(self.weight_decay, p=self.p).to(device)
            loss_reg = reg_loss(model.fc1) + reg_loss(model.fc2) + reg_loss(model.fc3)
            loss = args.gama * loss_cro + loss_reg + args.alpha * L / batch
        return loss


if __name__ == '__main__':

    main()

