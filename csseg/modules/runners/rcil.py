'''
Function:
    Implementation of "Representation Compensation Networks for Continual Semantic Segmentation"
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
from .base import BaseRunner


'''RCILRunner'''
class RCILRunner(BaseRunner):
    def __init__(self, mode, cmd_args, runner_cfg):
        super(MIBRunner, self).__init__(
            mode=mode, cmd_args=cmd_args, runner_cfg=runner_cfg
        )