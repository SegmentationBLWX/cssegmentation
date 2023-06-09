'''vocaug_512x512'''
import os


'''DATASET_CFG_VOCAUG_512x512'''
DATASET_CFG_VOCAUG_512x512 = {
    'type': 'VOCDataset',
    'rootdir': os.path.join(os.getcwd(), 'VOCdevkit/VOC2012'),
    'overlap': True, 
    'masking_value': 0,
    'train': {
        'set': 'trainaug',
        'transforms': [
            ('Resize', {'output_size': (2048, 512), 'keep_ratio': True, 'scale_range': (0.5, 2.0)}),
            ('RandomCrop', {'crop_size': (512, 512), 'one_category_max_ratio': 0.75}),
            ('RandomFlip', {'flip_prob': 0.5}),
            ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
            ('ToTensor', {}),
            ('Padding', {'output_size': (512, 512), 'data_type': 'tensor'}),
        ],
    },
    'test': {
        'set': 'val',
        'transforms': [
            ('Resize', {'output_size': (2048, 512), 'keep_ratio': True, 'scale_range': None}),
            ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
            ('ToTensor', {}),
        ],
    },
}