import sys, os, time, math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Input labels to one hot vector
def label2onehot(labels, pad_value):
    inp_ = torch.unsqueeze(labels, 2)
    one_hot = torch.FloatTensor(labels.size(0), labels.size(1), pad_value + 1).zero_().to(device)
    one_hot.scatter_(2, inp_, 1)
    one_hot, _ = one_hot.max(dim=1)
    one_hot = one_hot[:, 1:-1]
    one_hot[:, 0] = 0

    return one_hot

# draw training curve
def draw_result(epoch, train_loss, val_loss):
    for t in ['loss', 'ingrt']:
        plt.clf()
        plt.plot(range(epoch),train_loss[t],'-',linewidth=3,label='Train loss')
        plt.plot(range(epoch), val_loss[t],'-',linewidth=3,label='Validation loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.grid(True)
        plt.legend()
        plt.savefig(t+'_Loss.png')

#  save trained model
def save_model(model, optimizer, checkpoints_dir, suff=''):
    if torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(), os.path.join(
            checkpoints_dir, 'model' + suff + '.ckpt'))

    else:
        torch.save(model.state_dict(), os.path.join(
            checkpoints_dir, 'model' + suff + '.ckpt'))

    torch.save(optimizer.state_dict(), os.path.join(
        checkpoints_dir, 'optim' + suff + '.ckpt'))

# count parameters of a model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# make directory
def make_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

# update error metrics
def update_error_types(error_types, y_pred, y_true):

    error_types['tp_i'] += (y_pred * y_true).sum(0).cpu().data.numpy()
    error_types['fp_i'] += (y_pred * (1-y_true)).sum(0).cpu().data.numpy()
    error_types['fn_i'] += ((1-y_pred) * y_true).sum(0).cpu().data.numpy()
    error_types['tn_i'] += ((1-y_pred) * (1-y_true)).sum(0).cpu().data.numpy()

    error_types['tp_all'] += (y_pred * y_true).sum().item()
    error_types['fp_all'] += (y_pred * (1-y_true)).sum().item()
    error_types['fn_all'] += ((1-y_pred) * y_true).sum().item()

# compute metrixs
def compute_metrics(ret_metrics, error_types, metric_names, eps=1e-10, weights=None):

    if 'accuracy' in metric_names:
        ret_metrics['accuracy'].append(np.mean((error_types['tp_i'] + error_types['tn_i']) / (error_types['tp_i'] + error_types['fp_i'] + error_types['fn_i'] + error_types['tn_i'])))

    if 'f1' in metric_names:
        pre = error_types['tp_i'] / (error_types['tp_i'] + error_types['fp_i'] + eps)
        rec = error_types['tp_i'] / (error_types['tp_i'] + error_types['fn_i'] + eps)
        pre = error_types['tp_all'] / (error_types['tp_all'] + error_types['fp_all'] + eps)
        rec = error_types['tp_all'] / (error_types['tp_all'] + error_types['fn_all'] + eps)
        f1 = 2*(pre * rec) / (pre + rec + eps)
        ret_metrics['f1'].append(f1)
