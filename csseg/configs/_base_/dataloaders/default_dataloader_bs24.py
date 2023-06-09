'''default_dataloader_bs24'''
import os


'''DATALOADER_CFG_BS24'''
DATALOADER_CFG_BS24 = {
    'total_bs_for_auto_check': 24,
    'train': {
        'batch_size_per_gpu': 6, 'num_workers_per_gpu': 2, 'shuffle': True, 'pin_memory': True, 'drop_last': True
    },
    'test': {
        'batch_size_per_gpu': 1, 'num_workers_per_gpu': 2, 'shuffle': False, 'pin_memory': True, 'drop_last': False
    },
}