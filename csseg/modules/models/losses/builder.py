'''
Function:
    Implementation of BuildLoss
Author:
    Zhenchao Jin
'''
import copy
from .mseloss import MSELoss
from .klloss import KLDivLoss
from .csloss import CosineSimilarityLoss
from .celoss import CrossEntropyLoss, MIBUnbiasedCrossEntropyLoss


'''BuildLoss'''
def BuildLoss(loss_cfg):
    loss_cfg = copy.deepcopy(loss_cfg)
    # supported loss functions
    supported_lossfuncs = {
        'MSELoss': MSELoss, 'KLDivLoss': KLDivLoss, 'CrossEntropyLoss': CrossEntropyLoss, 'CosineSimilarityLoss': CosineSimilarityLoss,
        'MIBUnbiasedCrossEntropyLoss': MIBUnbiasedCrossEntropyLoss,
    }
    # parse
    loss_type = loss_cfg.pop('type')
    loss_func = supported_lossfuncs[loss_type](**loss_cfg)
    # return
    return loss_func