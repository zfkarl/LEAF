from __future__ import print_function
import os
#os.environ['CUDA_VISIBLE_DEVICES']='0'
import argparse
import shutil
import time
import random
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

from models.backbone import ResNet_18,ViT_base16,ViT_base16_weights
import dataset.raf as raf
import dataset.ferplus as ferplus
import dataset.affectnet as affectnet
from losses import SupConLoss
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from sklearn.metrics import balanced_accuracy_score

parser = argparse.ArgumentParser(description='PyTorch MixMatch Training')
# Optimization options
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=16, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--num_workers', type=int, default=12,
                        help='num of workers to use')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('--manualSeed', type=int, default=5, help='manual seed')
#Device options
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
#Method options
parser.add_argument('--n_labeled', type=int, default=98,
                        help='Number of labeled data')
parser.add_argument('--train_iteration', type=int, default=1600,
                        help='Number of iteration per epoch')
parser.add_argument('--out', default='result',
                        help='Directory to output the result')
parser.add_argument('--ema-decay', default=0.999, type=float)
#Data
parser.add_argument('--dataset', type=str, default='rafdb',
                        help="root path to train data directory")



args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy
best_balanced_acc = 0
def main():
    global best_acc
    global best_balanced_acc

    path1 = os.path.join(args.dataset,str(args.n_labeled))
    path = os.path.join(args.out,path1)
    
    if not os.path.isdir(path):
        mkdir_p(path)

    # Data
    print(f'==> Preparing dataset')
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomApply([
            transforms.RandomCrop(224, padding=8)
        ], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    if args.dataset =='rafdb':
        FER_TestlabelPath = "/home/zf/dataset/FaceData/RAF-DB/label/test.txt"
        FER_TrainlabelPath = "/home/zf/dataset/FaceData/RAF-DB/label/train.txt"
        FER_img_folder_path ="/home/zf/dataset/FaceData/RAF-DB/aligned/" 
        train_labeled_set, train_unlabeled_set, test_set = raf.get_raf(FER_img_folder_path,
            FER_TrainlabelPath,FER_TestlabelPath, args.n_labeled, transform_train=transform_train, transform_val=transform_val)

        labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
        unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
        test_loader = data.DataLoader(test_set, batch_size=16, shuffle=False, num_workers=args.num_workers)

    elif args.dataset == 'ferplus':
        train_labeled_set, train_unlabeled_set, test_set = ferplus.get_ferplus(args.n_labeled, transform_train=transform_train, transform_val=transform_val)

        labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
        unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
        test_loader = data.DataLoader(test_set, batch_size=20, shuffle=False, num_workers=args.num_workers)
    
    elif args.dataset == 'affectnet7':
        train_labeled_set, train_unlabeled_set, test_set = affectnet.get_affectnet7(args.n_labeled, transform_train=transform_train, transform_val=transform_val)

        labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
        unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
        test_loader = data.DataLoader(test_set, batch_size=20, shuffle=False, num_workers=args.num_workers)
        
    elif args.dataset == 'affectnet8':
        train_labeled_set, train_unlabeled_set, test_set = affectnet.get_affectnet8(args.n_labeled, transform_train=transform_train, transform_val=transform_val)

        labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
        unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
        test_loader = data.DataLoader(test_set, batch_size=20, shuffle=False, num_workers=args.num_workers)  
               
    # Model
    print("==> creating ResNet-18")

    def create_model(ema=False):
        if args.dataset =='ferplus' or args.dataset =='affectnet8':
            num_classes = 8 
        else:
            num_classes = 7
            
            
        model = ResNet_18(num_classes)
        #model = ViT_base16(num_classes=7)
        #model = ViT_base16_weights(num_classes=7)
        
        model = torch.nn.DataParallel(model).cuda()

        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.CrossEntropyLoss(reduction='none')
    criterion_simclr = SupConLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr*args.batch_size/16)

    ema_optimizer= WeightEMA(model, ema_model, alpha=args.ema_decay)

    logger = Logger(os.path.join(path, 'log.txt'), title='dataset')
    logger.set_names(['Train Loss', 'Train Loss X', 'Train Loss U', 'Train Loss S', 'Test Loss', 'Test Acc.'])

    test_accs = []
    if args.dataset =='ferplus' or args.dataset =='affectnet8':
        num_classes = 8 
    else:
        num_classes = 7
    if num_classes == 7:
        threshold = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
    else:
        threshold = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,0.8]
    start_epoch = 1
    # Train and val
    for epoch in range(start_epoch, args.epochs + 1):
        if num_classes == 7:
            print('\nEpoch: [%d | %d] LR: %f Threshold=[%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]' % (epoch, args.epochs, state['lr'], threshold[0], threshold[1], threshold[2], threshold[3], threshold[4], threshold[5], threshold[6]))
        else:
            print('\nEpoch: [%d | %d] LR: %f Threshold=[%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]' % (epoch, args.epochs, state['lr'], threshold[0], threshold[1], threshold[2], threshold[3], threshold[4], threshold[5], threshold[6], threshold[7]))
        train_loss, train_loss_x, train_loss_u, train_loss_sim = train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion, criterion_simclr, threshold, epoch, use_cuda)
        _, train_acc, balanced_top1,outputs_new, targets_new = validate(labeled_trainloader, ema_model, criterion, epoch, use_cuda, mode='Train Stats')
        threshold = adaptive_threshold_generate(outputs_new, targets_new, threshold, epoch)

        test_loss, test_acc, balanced_top1, _ ,_= validate(test_loader, ema_model, criterion, epoch, use_cuda, mode='Test Stats')

        # append logger file
        logger.append([train_loss, train_loss_x, train_loss_u, train_loss_sim, test_loss, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        best_balanced_acc = max(balanced_top1, best_balanced_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'best_balanced_acc':  best_balanced_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best)
        test_accs.append(test_acc)
    print('Best acc:')
    print(round(best_acc,2))
    print('Best balanced acc:')
    print(round(best_balanced_acc*100,2))
    
    logger.append(['BatchSize:',args.batch_size,'Iteration:',args.train_iteration,'Best Acc:',round(best_acc,2)])
    logger.append(['Epoch:',args.epochs,'LearningRate:',args.lr,'Best Balanced Acc:',round(best_balanced_acc*100,2)])
    logger.close()

def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion_ce, criterion_simclr, threshold, epoch, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    losses_sim = AverageMeter()
    end = time.time()

    bar = Bar('Training', max=args.train_iteration)
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    model.train()
    for batch_idx in range(args.train_iteration):
        try:
            inputs_x, targets_x = next(labeled_train_iter)
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x = next(labeled_train_iter)

        try:
            (inputs_u, inputs_u2, inputs_strong), _ = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2, inputs_strong), _ = next(unlabeled_train_iter)

        # measure data loading time
        data_time.update(time.time() - end)

        batch_size = inputs_x.size(0)


        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u = inputs_u.cuda()
            inputs_u2 = inputs_u2.cuda()
            inputs_strong = inputs_strong.cuda()

        # compute guessed labels of unlabeled samples
        outputs_u, feature_u = model(inputs_u)
        
        outputs_u2, feature_u2 = model(inputs_u2)
        
            
        p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
        max_probs, max_idx = torch.max(p, dim=1)
        max_idx = max_idx.detach()

        output_x, _ = model(inputs_x)
            
        num_classes = output_x.shape[-1]
        mask = mask_generate(num_classes,max_probs, max_idx, batch_size, threshold)
        mask_idx = np.where(mask.cpu() == 0)[0]
        features_prob = torch.cat([feature_u[mask_idx, :].unsqueeze(1), feature_u2[mask_idx, :].unsqueeze(1)], dim=1)

        #print('Targets_x:',targets_x)
        targets_x = F.one_hot(targets_x.squeeze(-1).to(torch.int64), num_classes=output_x.shape[-1]).to(torch.float32)
        #print('Targets_x:',targets_x.shape)
        #print('Output_x:',output_x.shape)
        Lx = criterion_ce(output_x, targets_x).mean()

        if features_prob.shape[0] ==0:
            Ls = torch.from_numpy(np.array(0))
        else:
            Ls = criterion_simclr(features_prob)

        output_strong, _ = model(inputs_strong)

        
        ### CE starts####
        # Lu = criterion_ce(output_strong, max_idx) * mask
        # Lu = Lu.mean()
        ### CE ends####
        
        # ###CPL starts####
        Lu_base = criterion_ce(output_strong, max_idx).mean()
        outputs_avg =(outputs_u+outputs_u2) / 2
        loss_topk, weight_mask = topk_ce_FPL(output_strong,outputs_avg)
        
        Lu = (loss_topk*weight_mask).mean() 
        
        Lu_weight_balance = Lu_base.item()/(Lu.item())
        Lu = Lu * Lu_weight_balance * mask
        Lu = Lu.mean()
        # ###CPL ends####
        
        loss = Lx *0.5 + Lu + Ls * 0.1 

        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0))
        losses_sim.update(Ls.item(), inputs_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Total: {total:} | Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f} | Loss_s: {loss_sim:.4f}'.format(
                    batch=batch_idx + 1,
                    size=args.train_iteration,
                    total=bar.elapsed_td,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    loss_sim=losses_sim.avg,
                    )
        bar.next()
    bar.finish()

    return (losses.avg, losses_x.avg, losses_u.avg, losses_sim.avg)

def validate(valloader, model, criterion, epoch, use_cuda, mode):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar(f'{mode}', max=len(valloader))
    
    if args.dataset =='ferplus' or args.dataset =='affectnet8':
        num_classes = 8 
    else:
        num_classes = 7
    outputs_new = torch.ones(1, num_classes).cuda()
    targets_new = torch.ones(1).long().cuda()

    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output
            outputs, _ = model(inputs)
            targets1 = F.one_hot(targets.squeeze(-1).to(torch.int64), num_classes=outputs.shape[-1]).to(torch.float32)
            # print('outputs: ', outputs.shape)
            # print('targets1: ', targets1.shape)
            loss = criterion(outputs, targets1).mean()

            ##
            outputs_new = torch.cat((outputs_new, outputs), dim=0)
            targets_new = torch.cat((targets_new, targets.squeeze(-1)), dim=0)
            ##
            y_true.extend(targets.cpu().tolist())
            y_pred.extend(torch.max(outputs, dim=-1)[1].cpu().tolist())
            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs, targets.squeeze(-1), topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Total: {total:} | Loss: {loss:.4f} | Accuracy: {top1: .4f} '.format(
                        batch=batch_idx + 1,
                        size=len(valloader),
                        total=bar.elapsed_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        
                        )
            bar.next()
        bar.finish()
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    balanced_top1 = balanced_accuracy_score(y_true, y_pred)
    return (losses.avg, top1.avg,balanced_top1, outputs_new, targets_new)

def save_checkpoint(state, is_best, checkpoint=args.out, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def linear_rampup(current, rampup_length=args.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype==torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                # customized weight decay
                param.mul_(1 - self.wd)

def mask_generate(num_classes, max_probs, max_idx, batch, threshold):
    mask_ori = torch.zeros(batch).cuda()
    for i in range(num_classes):
        idx = np.where(max_idx.cpu() == i)[0]
        #print(idx)
        m = max_probs[idx].ge(threshold[i]).float()
        for k in range(len(idx)):
            mask_ori[idx[k]]+=m[k]
    return mask_ori.cuda()

def adaptive_threshold_generate(outputs, targets, threshold, epoch):
    outputs_l = outputs[1:, :]
    targets_l = targets[1:]
    probs = torch.softmax(outputs_l, dim=1)
    max_probs, max_idx = torch.max(probs, dim=1)
    eq_idx = np.where(targets_l.eq(max_idx).cpu() == 1)[0]

    probs_new = max_probs[eq_idx]
    targets_new = targets_l[eq_idx]
    
    num_classes = outputs.shape[-1]
    
    for i in range(num_classes):
        idx = np.where(targets_new.cpu() == i)[0]
        if idx.shape[0] != 0:
            threshold[i] = probs_new[idx].mean().cpu() * 0.97 / (1 + math.exp(-1 * epoch)) if probs_new[idx].mean().cpu() * 0.97 / (1 + math.exp(-1 * epoch)) >= 0.8 else 0.8
        else:
            threshold[i] = 0.8
    return threshold

def to_one_hot(tensor,nClasses):
    one_hot = torch.nn.functional.one_hot(tensor, nClasses)
    return one_hot


def topk_ce_FPL(pred, inputs_2, temp_k=3, threshold=0.90, weight='concave'):
    soft_plus = torch.nn.Softplus()
    prob_topk, pse_topk = torch.topk(inputs_2.float().softmax(dim=1).detach(), k=temp_k + 1, dim=1)
    class_num = pred.shape[1]

    sum_pred_pos = torch.zeros_like(pred) - 1  # b,c,w,h
    sum_pred_neg = torch.zeros_like(pred) - 1  # b,c,w,h
    flag_mask = torch.zeros_like(pred[:, 0]) - 1  # b,w,h
    weight_mask = torch.zeros_like(pred[:, 0]) + 1
    one_hot_dict = {}
    for i in range(pse_topk.shape[1]):
        one_hot_dict[i] = to_one_hot(pse_topk[:, i], class_num)

    cumulative_prob = prob_topk.clone()
    for i in range(1, cumulative_prob.shape[1]):
        cumulative_prob[:, i] = cumulative_prob[:, i] + cumulative_prob[:, i - 1]

    for i in range(1, cumulative_prob.shape[1]):
        k = i - 1  # i-1 form 0
        if i == cumulative_prob.shape[1] - 1:
            mask_k = (flag_mask == -1)
        else:
            mask_k = (cumulative_prob[:, i] >= threshold) * (flag_mask == -1)
        flag_mask[mask_k] = 1

        if weight == 'linear':
            weight_mask[mask_k] = ((cumulative_prob[:, i] / (i + 1) - prob_topk[:, i]) / (
                        cumulative_prob[:, i] / ((i + 1))))[mask_k]
        elif weight == 'convex':
            weight_mask[mask_k] = ((cumulative_prob[:, i] / (i + 1) - prob_topk[:, i]) / (
                        prob_topk[:, i] + cumulative_prob[:, i] / (i + 1)))[mask_k]
        elif weight == 'concave':
            weight_mask[mask_k] = (torch.log(
                1 + cumulative_prob[:, i] / (i + 1) * 50 - prob_topk[:, i] * 50) / torch.log(
                1 + 50 * cumulative_prob[:, i] / ((i + 1))))[mask_k]

        mask_k = mask_k[:, None].expand(pred.shape[0], class_num).contiguous()

        pse_mask_pos = 0
        for j in range(k + 1):
            pse_mask_pos += one_hot_dict[j]  # b,c,w,h
        pse_mask_neg = 1 - pse_mask_pos
        sum_pred_pos[mask_k] = (-(pred * pse_mask_pos) - pse_mask_neg * 1e7)[mask_k]
        sum_pred_neg[mask_k] = ((pred * pse_mask_neg) - pse_mask_pos * 1e7)[mask_k]

    loss_topk = soft_plus(torch.logsumexp(sum_pred_pos, dim=1) + torch.logsumexp(sum_pred_neg, dim=1))
    weight_mask[weight_mask > 1] = 1
    return loss_topk, weight_mask

# def norm(filt):
#     # filt (dim, out_dim)
#     filt_norm = ((filt * filt).sum(dim=0) + 1e-8).sqrt()
#     filt_norm = filt_norm.reshape(1, filt.shape[1])
#     return filt / filt_norm

# def cal(filt):
#     filt_norm = ((filt * filt).sum(dim=0) + 1e-8).sqrt()
#     filt_norm = filt_norm.reshape(1, filt.shape[1])
#     norm_mat = torch.matmul(filt_norm.transpose(1,0), filt_norm)
#     inner_pro = torch.matmul(filt.transpose(1,0), filt)
#     return inner_pro / norm_mat


# def loss_mhs_weight_reg(filt):
#     # filt (output_dim, input_dim)
#     filt = filt.transpose(1,0) # (in, out)
#     filt = norm(filt)
#     inner_pro = cal(filt)
#     final = (2.0 - 2.0 * inner_pro)
#     final -= torch.triu(final)
#     nonzeros = torch.where(final!=0)
#     if torch.all(final == 0):
#         target = torch.zeros(1,1).to(final.device)
#     else:
#         target = torch.min(final[nonzeros])
#     #print(target)
#     mask = final.eq(target)
#     loss = -(final * mask.detach()).sum()

#     return loss

if __name__ == '__main__':
    main()
