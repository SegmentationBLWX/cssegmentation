'''
Function:
    Implementation of DatasetBuilder and BuildDataset
Author:
    Zhenchao Jin
'''
import copy
from .voc import VOCDataset
from .ade20k import ADE20kDataset
from ..utils import BaseModuleBuilder


'''DatasetBuilder'''
class DatasetBuilder(BaseModuleBuilder):
    SUPPORTED_MODULES = {
        'VOCDataset': VOCDataset, 'ADE20kDataset': ADE20kDataset,
    }
    '''build'''
    def build(self, mode, task_name, task_id, dataset_cfg):
        dataset_cfg = copy.deepcopy(dataset_cfg)
        train_cfg, test_cfg = dataset_cfg.pop('train'), dataset_cfg.pop('test')
        dataset_cfg.update(train_cfg if mode == 'TRAIN' else test_cfg)
        dataset_type = dataset_cfg.pop('type')
        module_cfg = {
            'mode': mode, 'task_name': task_name, 'task_id': task_id, 'dataset_cfg': dataset_cfg, 'type': dataset_type
        }
        return super().build(module_cfg)


'''BuildDataset'''
BuildDataset = DatasetBuilder().build