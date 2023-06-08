'''
Function:
    Implementation of CosineSimilarityLoss
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


'''CosineSimilarityLoss'''
class CosineSimilarityLoss(nn.Module):
    def __init__(self, scale_factor=1.0, reduction='mean'):
        super(CosineSimilarityLoss, self).__init__()
        self.reduction = reduction
        self.scale_factor = scale_factor
    '''forward'''
    @torch.autocast()
    def forward(self, prediction, target):
        # assert
        assert prediction.size() == target.size()
        # calculate loss according to config
        loss = 1 - F.cosine_similarity(prediction, target, dim=1)
        if self.reduction == 'mean': 
            loss = loss.mean()
        elif self.reduction == 'sum': 
            loss = loss.sum()
        loss = loss * self.scale_factor
        # return
        return loss