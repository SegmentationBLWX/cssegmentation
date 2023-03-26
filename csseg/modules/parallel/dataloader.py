'''
Function:
    Implementation of BuildDistributedDataloader
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.utils.data.distributed.DistributedSampler as DistributedSampler


'''BuildDistributedDataloader'''
def BuildDistributedDataloader(dataset, dataloader_cfg):
    dataloader_cfg = copy.deepcopy(dataloader_cfg)
    # parse
    dataloader_cfg = dataloader_cfg[dataset.mode.lower()]
    shuffle = dataloader_cfg.pop('shuffle')
    dataloader_cfg['shuffle'] = False
    # sampler
    sampler = DistributedSampler(dataset, shuffle=shuffle)
    dataloader_cfg['sampler'] = sampler
    # dataloader
    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_cfg)
    # return
    return dataloader