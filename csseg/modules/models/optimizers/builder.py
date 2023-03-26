'''
Function:
    Implementation of BuildOptimizer
Author:
    Zhenchao Jin
'''
import copy
import torch
from .constructors import DefaultParamsConstructor


'''BuildOptimizer'''
def BuildOptimizer(model, optimizer_cfg):
    optimizer_cfg = copy.deepcopy(optimizer_cfg)
    constructor_cfg = optimizer_cfg.pop('constructor_cfg')
    # supported optimizers
    supported_optimizers = {
        'SGD': torch.optim.SGD,
        'Adam': torch.optim.Adam,
        'Rprop': torch.optim.Rprop,
        'AdamW': torch.optim.AdamW,
        'RMSprop': torch.optim.RMSprop,
        'Adagrad': torch.optim.Adagrad,
        'Adadelta': torch.optim.Adadelta,
    }
    # parse
    optimizer_type = optimizer_cfg.pop('type')
    constructor_type = constructor_cfg.pop('type')
    supported_paramsconstructors = {
        'DefaultParamsConstructor': DefaultParamsConstructor
    }
    params_constructor = supported_paramsconstructors[constructor_type](optimizer_cfg, **constructor_cfg)
    optimizer_cfg['params'] = params_constructor(model)
    optimizer = supported_optimizers[optimizer_type](**optimizer_cfg)
    # return
    return optimizer