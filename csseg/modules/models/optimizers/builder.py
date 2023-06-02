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
    params_constructor_cfg = optimizer_cfg.pop('params_constructor_cfg')
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
    params_constructor_type = params_constructor_cfg.pop('type')
    supported_paramsconstructors = {
        'DefaultParamsConstructor': DefaultParamsConstructor
    }
    params_constructor = supported_paramsconstructors[params_constructor_type](optimizer_cfg, **params_constructor_cfg)
    optimizer_cfg['params'] = params_constructor(model)
    optimizer = supported_optimizers[optimizer_type](**optimizer_cfg)
    # return
    return optimizer