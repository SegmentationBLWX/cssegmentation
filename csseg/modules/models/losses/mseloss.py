'''
Function:
    Implementation of MSELoss
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


'''MSELoss'''
class MSELoss(nn.Module):
    def __init__(self, scale_factor=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        self.reduction = reduction
        self.scale_factor = scale_factor
    '''forward'''
    def forward(self, prediction, target):
        # assert
        assert prediction.size() == target.size()
        # calculate loss according to config
        loss = F.mse_loss(prediction, target, reduction=self.reduction)
        loss = loss * self.scale_factor
        # return
        return loss