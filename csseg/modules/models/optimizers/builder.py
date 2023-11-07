'''
Function:
    Implementation of BuildOptimizer and OptimizerBuilder
Author:
    Zhenchao Jin
'''
import copy
import torch.optim as optim
from ...utils import BaseModuleBuilder
from .paramsconstructor import BuildParamsConstructor


'''OptimizerBuilder'''
class OptimizerBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'SGD': optim.SGD, 'Adam': optim.Adam, 'AdamW': optim.AdamW, 'Adadelta': optim.Adadelta,
        'Adagrad': optim.Adagrad, 'Rprop': optim.Rprop, 'RMSprop': optim.RMSprop, 
    }
    '''build'''
    def build(self, model, optimizer_cfg):
        # parse config
        optimizer_cfg = copy.deepcopy(optimizer_cfg)
        optimizer_type = optimizer_cfg.pop('type')
        paramwise_cfg, filter_params = optimizer_cfg.pop('paramwise_cfg', {}), optimizer_cfg.pop('filter_params', False)
        # build params_constructor
        params_constructor = BuildParamsConstructor(paramwise_cfg=paramwise_cfg, filter_params=filter_params, optimizer_cfg=optimizer_cfg)
        # obtain params
        optimizer_cfg['params'] = params_constructor(model=model)
        # build optimizer
        optimizer = self.REGISTERED_MODULES[optimizer_type](**optimizer_cfg)
        # return
        return optimizer


'''BuildOptimizer'''
BuildOptimizer = OptimizerBuilder().build