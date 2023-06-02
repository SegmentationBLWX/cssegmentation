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