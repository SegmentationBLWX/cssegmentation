'''
Function:
    Implementation of BuildLoss
Author:
    Zhenchao Jin
'''
import copy
from .klloss import KLDivLoss
from .celoss import CrossEntropyLoss


'''BuildLoss'''
def BuildLoss(loss_cfg):
    loss_cfg = copy.deepcopy(loss_cfg)
    # supported loss functions
    supported_lossfuncs = {
        'KLDivLoss': KLDivLoss,
        'CrossEntropyLoss': CrossEntropyLoss,
    }
    # parse
    loss_type = loss_cfg.pop('type')
    loss_func = supported_lossfuncs[loss_type](**loss_cfg)
    # return
    return loss_func