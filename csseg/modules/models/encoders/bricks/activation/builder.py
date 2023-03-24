'''
Function:
    Implementation of BuildActivation
Author:
    Zhenchao Jin
'''
import copy
import torch.nn as nn


'''BuildActivation'''
def BuildActivation(act_cfg):
    if act_cfg is None: return nn.Identity()
    act_cfg = copy.deepcopy(act_cfg)
    # supported activations
    supported_activations = {
        'Tanh': nn.Tanh,
        'ReLU': nn.ReLU,
        'GELU': nn.GELU,
        'SELU': nn.SELU,
        'ReLU6': nn.ReLU6,
        'PReLU': nn.PReLU,
        'Sigmoid': nn.Sigmoid,
        'LeakyReLU': nn.LeakyReLU,
    }
    # parse
    act_type = act_cfg.pop('type')
    act_layer = supported_activations[act_type](**act_cfg)
    # return
    return act_layer