'''default_parallel'''
import os


'''PARALLEL_CFG'''
PARALLEL_CFG = {
    'backend': 'nccl', 'init_method': 'env://', 'opt_level': ['O0', 'O1', 'O2', 'O3'][1],
}