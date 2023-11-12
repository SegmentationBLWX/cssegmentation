'''default_dataloader_bs24'''
import os


'''DATALOADER_CFG_BS24'''
DATALOADER_CFG_BS24 = {
    'total_train_bs_for_auto_check': 24,
    'auto_align_train_bs': True,
    'train': {
        'batch_size_per_gpu': 12, 'num_workers_per_gpu': 4, 'shuffle': True, 'pin_memory': True, 'drop_last': True
    },
    'test': {
        'batch_size_per_gpu': 12, 'num_workers_per_gpu': 4, 'shuffle': False, 'pin_memory': True, 'drop_last': False
    },
}