'''default_parallel'''
import os


'''PARALLEL_CFG'''
PARALLEL_CFG = {
    'backend': 'nccl', 'init_method': 'env://', 'parallel_model_cfg': {}, 'grad_scaler_cfg': {},
}