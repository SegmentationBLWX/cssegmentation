'''
Function:
    Implementation of BuildNormalization
Author:
    Zhenchao Jin
'''
import copy
import torch.nn as nn
from inplace_abn import ABN, InPlaceABN, InPlaceABNSync


'''BuildNormalization'''
def BuildNormalization(placeholder, norm_cfg):
    if norm_cfg is None: return nn.Identity()
    norm_cfg = copy.deepcopy(norm_cfg)
    # supported normalizations
    supported_normalizations = {
        'ABN': ABN,
        'InPlaceABN': InPlaceABN,
        'LayerNorm': nn.LayerNorm,
        'GroupNorm': nn.GroupNorm,
        'BatchNorm1d': nn.BatchNorm1d,
        'BatchNorm2d': nn.BatchNorm2d,
        'BatchNorm3d': nn.BatchNorm3d,
        'InPlaceABNSync': InPlaceABNSync,
        'SyncBatchNorm': nn.SyncBatchNorm,
        'InstanceNorm1d': nn.InstanceNorm1d,
        'InstanceNorm2d': nn.InstanceNorm2d,
        'InstanceNorm3d': nn.InstanceNorm3d,
    }
    # parse
    norm_type = norm_cfg.pop('type')
    norm_layer = supported_normalizations[norm_type](placeholder, **norm_cfg)
    # return
    return norm_layer