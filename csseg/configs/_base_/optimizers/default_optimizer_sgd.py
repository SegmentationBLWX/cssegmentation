'''default_optimizer_sgd'''
import os


'''OPTIMIZER_CFG_SGD'''
OPTIMIZER_CFG_SGD = {
    'type': 'SGD', 'momentum': 0.9, 'nesterov': True, 'weight_decay': 1e-4, 'lr': None,
    'params_constructor_cfg': {'type': 'DefaultParamsConstructor', 'filter_params': True, 'paramwise_cfg': None},
}