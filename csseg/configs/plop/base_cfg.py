'''BASE_CFG for PLOP'''
# DATASET_CFG
DATASET_CFG = {
    'type': '',
    'rootdir': '', 
    'overlap': True, 
    'masking_value': 0,
    'train': {
        'set': 'train',
        'transforms': [
            ('RandomResizedCrop', {'output_size': 512, 'scale': (0.5, 2.0)}),
            ('RandomHorizontalFlip', {}),
            ('ToTensor', {}),
            ('Normalize', {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}),
        ],
    },
    'test': {
        'set': 'val',
        'transforms': [
            ('Resize', {'output_size': 512}),
            ('CenterCrop', {'output_size': 512}),
            ('ToTensor', {}),
            ('Normalize', {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}),
        ],
    },
}
# DATALOADER_CFG
DATALOADER_CFG = {
    'train': {'batch_size': 12, 'num_workers': 1, 'shuffle': True, 'pin_memory': True, 'drop_last': True},
    'test': {'batch_size': 12, 'num_workers': 1, 'shuffle': False, 'pin_memory': True, 'drop_last': False},
}
# SEGMENTOR_CFG
SEGMENTOR_CFG = {
    'type': 'PLOPSegmentor',
    'num_known_classes_list': None,
    'selected_indices': (3,), 
    'align_corners': False, 
    'encoder_cfg': {
        'type': 'ResNetPLOP',
        'depth': 101,
        'outstride': 16,
        'out_indices': (0, 1, 2, 3),
        'norm_cfg': {'type': 'InPlaceABNSync', 'activation': 'leaky_relu', 'activation_param': 0.01},
        'act_cfg': None,
        'pretrained': True,
    }, 
    'decoder_cfg': {
        'type': 'ASPPHead',
        'in_channels': 2048,
        'out_channels': 256,
        'dilations': (1, 6, 12, 18),
        'align_corners': False,
        'norm_cfg': {'type': 'InPlaceABNSync', 'activation': 'leaky_relu', 'activation_param': 0.01},
        'act_cfg': None,
    },
}
# OPTIMIZER_CFGS
OPTIMIZER_CFGS = [{
    'constructor_cfg': {'type': 'DefaultParamsConstructor', 'filter_params': True, 'paramwise_cfg': None},
    'type': 'SGD',
    'momentum': 0.9, 
    'nesterov': True,
    'weight_decay': 1e-4,
}]
# SCHEDULER_CFGS
SCHEDULER_CFGS = [{
    'type': 'PolyScheduler',
    'max_iters': -1, 
    'max_epochs': -1, 
    'lr': 0.01, 
    'min_lr': 0.0, 
    'power': 0.9
}]
# PARALLEL_CFG
PARALLEL_CFG = {
    'backend': 'nccl',
    'init_method': 'env://',
    'opt_level': ['O0', 'O1', 'O2', 'O3'][1],
}
# LOSSES_CFGS
LOSSES_CFGS = {
    'segmentation': {'loss_seg': {'CrossEntropyLoss': {'scale_factor': None, 'reduction': 'none', 'ignore_index': 255}}},
    'distillation': {'pod_factor': 0.01, 'pod_factor_last_scale': 0.0005, 'spp_scales': [1, 2, 4]},
}
# RUNNER_CFG
RUNNER_CFG = {
    'DATASET_CFG': DATASET_CFG, 'DATALOADER_CFG': DATALOADER_CFG, 'SEGMENTOR_CFG': SEGMENTOR_CFG,
    'OPTIMIZER_CFGS': OPTIMIZER_CFGS, 'SCHEDULER_CFGS': SCHEDULER_CFGS, 'PARALLEL_CFG': PARALLEL_CFG,
    'LOSSES_CFGS': LOSSES_CFGS,
}
RUNNER_CFG.update({
    'type': 'PLOPRunner',
    'algorithm': 'PLOP',
    'task_name': '',
    'task_id': -1,
    'num_tasks': -1,
    'work_dir': '',
    'save_interval_epochs': 1,
    'eval_interval_epochs': 1,
    'log_interval_iterations': 50,
    'choose_best_segmentor_by_metric': 'mean_iou',
    'logfilepath': '',
    'num_total_classes': -1,
    'pseudolabeling_minimal_threshold': 0.001,
    'random_seed': 42,
})