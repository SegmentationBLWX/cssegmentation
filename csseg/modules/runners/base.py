'''
Function:
    Implementation of BaseRunner
Author:
    Zhenchao Jin
'''
from ..utils import Logger
from ..datasets import BuildDataset
from ..models import BuildSegmentor, BuildLoss
from ..parallel import BuildDistributedDataloader


'''BaseRunner'''
class BaseRunner():
    def __init__(self, cmd_args, runner_cfg):
        # build logger handle
        self.logger_handle = Logger(logfilepath=runner_cfg['logfilepath'])
        # build datasets
        dataset_cfg = runner_cfg['DATASET_CFG']
        train_set = BuildDataset(mode='TRAIN', dataset_cfg=dataset_cfg)
        test_set = BuildDataset(mode='TEST', dataset_cfg=dataset_cfg)
        # build dataloaders
        dataloader_cfg = runner_cfg['DATALOADER_CFG']
        self.train_loader = BuildDistributedDataloader(train_set, dataloader_cfg)
        self.test_loader = BuildDistributedDataloader(test_set, dataloader_cfg)
        # build segmentor
        segmentor_cfg = runner_cfg['SEGMENTOR_CFG']
        self.segmentor = BuildSegmentor(segmentor_cfg)
    '''start'''
    def start(self):
        pass
