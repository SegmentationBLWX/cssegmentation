'''
Function:
    Implementation of BuildDistributedModel
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
try:
    from apex.parallel import DistributedDataParallel
except:
    DistributedDataParallel = None


'''BuildDistributedModel'''
def BuildDistributedModel(model, model_cfg):
    if DistributedDataParallel is None:
        return nn.parallel.DistributedDataParallel(model, **model_cfg)
    else:
        return DistributedDataParallel(model, **model_cfg)