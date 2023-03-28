'''plop_r101iabnd16_aspp_512x512_vocaug15-1_overlap'''
import os
from .base_cfg import RUNNER_CFG


# modify DATASET_CFG
RUNNER_CFG['DATASET_CFG'].update({
    'type': 'VOCDataset',
    'rootdir': os.path.join(os.getcwd(), 'VOCdevkit/VOC2012'),
    'overlap': True,
})
RUNNER_CFG['DATASET_CFG']['train']['set'] = 'trainaug'
# modify SCHEDULER_CFGS
RUNNER_CFG['SCHEDULER_CFGS'] = [
    {'type': 'PolyScheduler', 'max_iters': -1, 'max_epochs': 30, 'lr': 0.01, 'min_lr': 0.0, 'power': 0.9},
    {'type': 'PolyScheduler', 'max_iters': -1, 'max_epochs': 30, 'lr': 0.001, 'min_lr': 0.0, 'power': 0.9},
    {'type': 'PolyScheduler', 'max_iters': -1, 'max_epochs': 30, 'lr': 0.001, 'min_lr': 0.0, 'power': 0.9},
    {'type': 'PolyScheduler', 'max_iters': -1, 'max_epochs': 30, 'lr': 0.001, 'min_lr': 0.0, 'power': 0.9},
    {'type': 'PolyScheduler', 'max_iters': -1, 'max_epochs': 30, 'lr': 0.001, 'min_lr': 0.0, 'power': 0.9},
    {'type': 'PolyScheduler', 'max_iters': -1, 'max_epochs': 30, 'lr': 0.001, 'min_lr': 0.0, 'power': 0.9},
]
# modify RUNNER_CFG
RUNNER_CFG.update({
    'task_name': '15-5s',
    'num_tasks': 6,
    'num_total_classes': 21,
    'work_dir': 'plop_r101iabnd16_aspp_512x512_vocaug15-1_overlap',
    'logfilepath': 'plop_r101iabnd16_aspp_512x512_vocaug15-1_overlap/plop_r101iabnd16_aspp_512x512_vocaug15-1_overlap.log',
})