'''default_parallel'''
import os


'''PARALLEL_CFG'''
PARALLEL_CFG = {
    'backend': 'nccl', 'init_method': 'env://', 'model_cfg': {}
}