'''
Function:
    Implementation of "Continual semantic segmentation via repulsion-attraction of sparse and disentangled latent representations"
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


'''SDRRunner'''
class SDRRunner(BaseRunner):
    def __init__(self, mode, cmd_args, runner_cfg):
        super(SDRRunner, self).__init__(
            mode=mode, cmd_args=cmd_args, runner_cfg=runner_cfg
        )