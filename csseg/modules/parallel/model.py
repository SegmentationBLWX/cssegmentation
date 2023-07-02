'''
Function:
    Implementation of BuildDistributedModel
Author:
    Zhenchao Jin
'''
from apex.parallel import DistributedDataParallel


'''BuildDistributedModel'''
def BuildDistributedModel(model, model_cfg):
    return DistributedDataParallel(model, **model_cfg)