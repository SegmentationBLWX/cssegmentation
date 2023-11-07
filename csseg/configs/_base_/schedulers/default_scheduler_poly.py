'''default_scheduler_poly'''
import os


'''SCHEDULER_CFG_POLY'''
SCHEDULER_CFG_POLY = {
    'type': 'PolyScheduler', 'iters_per_epoch': -1, 'max_epochs': -1, 'lr': 0.01, 'min_lr': 0.0, 'power': 0.9,
    'optimizer': {
        'type': 'SGD', 'momentum': 0.9, 'nesterov': True, 'weight_decay': 1e-4, 'lr': None,
        'paramwise_cfg': {'type': 'DefaultParamsConstructor'}, 'filter_params': True,
    }
}