'''
Function:
    Implementation of KLDivLoss
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


'''KLDivLoss'''
class KLDivLoss(nn.Module):
    def __init__(self, log_target=None, reduction='mean', temperature=1.0, scale_factor=1.0):
        super(KLDivLoss, self).__init__()
        self.reduction = reduction
        self.log_target = log_target
        self.temperature = temperature
        self.scale_factor = scale_factor
    '''forward'''
    @torch.autocast()
    def forward(self, prediction, target):
        # assert
        assert prediction.size() == target.size()
        # construct config
        kl_args = {
            'reduction': self.reduction, 
        }
        if self.log_target is not None:
            kl_args.update({'log_target': self.log_target})
        # calculate loss according to config
        src_distribution = nn.LogSoftmax(dim=1)(prediction / self.temperature)
        tgt_distribution = nn.Softmax(dim=1)(target / self.temperature)
        loss = (self.temperature ** 2) * nn.KLDivLoss(**kl_args)(src_distribution, tgt_distribution)
        loss = loss * self.scale_factor
        # return
        return loss