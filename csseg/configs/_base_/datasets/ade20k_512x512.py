'''ade20k_512x512'''
import os


'''DATASET_CFG_ADE20K_512x512'''
DATASET_CFG_ADE20K_512x512 = {
    'type': 'ADE20kDataset',
    'rootdir': os.path.join(os.getcwd(), 'ADE20k'),
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