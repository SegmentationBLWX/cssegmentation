'''
Function:
    Implementation of ParamsConstructor
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn


'''DefaultParamsConstructor'''
class DefaultParamsConstructor():
    def __init__(self, optimizer_cfg, filter_params=True, paramwise_cfg=None):
        self.base_lr = optimizer_cfg['lr']
        self.base_weight_decay = optimizer_cfg['weight_decay']
        self.filter_params = filter_params
        self.paramwise_cfg = paramwise_cfg
    '''call'''
    def __call__(self, model):
        if hasattr(model, 'module'): model = model.module
        # not using tricks during training
        if self.paramwise_cfg is None:
            if self.filter_params:
                param_groups = filter(lambda p: p.requires_grad, model.parameters())
            else:
                param_groups = model.parameters()
        # using tricks during training
        else:
            param_groups = []
            for key, value in model.architectures():
                param_group = {
                    'name': key,
                    'lr': self.base_lr * self.paramwise_cfg[key]['lr_scale'],
                    'lr_scale': self.paramwise_cfg[key]['lr_scale'],
                    'weight_decay': self.base_weight_decay * self.paramwise_cfg[key]['weight_decay_scale'],
                    'weight_decay_scale': self.paramwise_cfg[key]['weight_decay_scale'],
                    'params': filter(lambda p: p.requires_grad, value.parameters()) if self.filter_params else value.parameters(),
                }
                param_groups.append(param_group)
        # return
        return param_groups