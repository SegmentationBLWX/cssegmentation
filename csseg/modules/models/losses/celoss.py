'''
Function:
    Implementation of CrossEntropyLoss
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


'''CrossEntropyLoss'''
class CrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean', ignore_index=255, scale_factor=1.0, weight=None, label_smoothing=None):
        super(CrossEntropyLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction
        self.scale_factor = scale_factor
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
    '''forward'''
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def forward(self, prediction, target):
        # construct config
        ce_args = {
            'weight': self.weight, 'ignore_index': self.ignore_index, 'reduction': self.reduction,
        }
        if self.label_smoothing is not None:
            ce_args.update({'label_smoothing': self.label_smoothing})
        # calculate loss according to config
        if prediction.dim() == target.dim():
            ce_args.pop('ignore_index')
            loss = F.cross_entropy(prediction, target, **ce_args)
        else:
            loss = F.cross_entropy(prediction, target.long(), **ce_args)
        loss = loss * self.scale_factor
        # return
        return loss


'''MIBUnbiasedCrossEntropyLoss'''
class MIBUnbiasedCrossEntropyLoss(nn.Module):
    def __init__(self, num_history_known_classes=None, reduction='mean', ignore_index=255, scale_factor=1.0):
        super(MIBUnbiasedCrossEntropyLoss, self).__init__()
        self.reduction = reduction
        self.scale_factor = scale_factor
        self.ignore_index = ignore_index
        self.num_history_known_classes = num_history_known_classes
    '''forward'''
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def forward(self, prediction, target):
        # calculate loss according to config
        num_history_known_classes = self.num_history_known_classes
        outputs = torch.zeros_like(prediction)
        den = torch.logsumexp(prediction, dim=1)
        outputs[:, 0] = torch.logsumexp(prediction[:, :num_history_known_classes], dim=1) - den
        outputs[:, num_history_known_classes:] = prediction[:, num_history_known_classes:] - den.unsqueeze(dim=1)
        labels = target.clone()
        labels[target < num_history_known_classes] = 0
        loss = F.nll_loss(outputs, labels, ignore_index=self.ignore_index, reduction=self.reduction)
        loss = loss * self.scale_factor
        # return
        return loss