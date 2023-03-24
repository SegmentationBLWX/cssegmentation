'''
Function:
    Implementation of BuildDataset
Author:
    Zhenchao Jin
'''
import copy
from .voc import VOCDataset
from .ade20k import ADE20kDataset


'''BuildDataset'''
def BuildDataset(mode, dataset_cfg):
    dataset_cfg = copy.deepcopy(dataset_cfg)
    # supported datasets
    supported_datasets = {
        'VOCDataset': VOCDataset,
        'ADE20kDataset': ADE20kDataset,
    }
    # parse
    train_cfg, test_cfg = dataset_cfg.pop('train'), dataset_cfg.pop('test')
    dataset_cfg.update(train_cfg if mode == 'TRAIN' else test_cfg)
    dataset_type = dataset_cfg.pop('type')
    dataset = supported_datasets[dataset_type](mode, dataset_cfg)
    # return
    return dataset