'''
Function:
    Implementation of "Class Similarity Weighted Knowledge Distillation for Continual Semantic Segmentation"
Author:
    Zhenchao Jin
'''
import copy
import torch
import functools
import torch.nn.functional as F
import torch.distributed as dist
from apex import amp
from .base import BaseRunner


'''CAFRunner'''
class CAFRunner(BaseRunner):
    def __init__(self, mode, cmd_args, runner_cfg):
        super(CAFRunner, self).__init__(
            mode=mode, cmd_args=cmd_args, runner_cfg=runner_cfg
        )