'''
Function:
    Implementation of BaseScheduler
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad


'''BaseScheduler'''
class BaseScheduler():
    def __init__(self, optimizer, max_iters=-1, max_epochs=-1, lr=0.01, min_lr=None):
        # set attributes
        self.lr = lr
        self.cur_lr = lr
        self.cur_iter = 0
        self.cur_epoch = 1
        self.optimizer = optimizer
        self.max_iters = max_iters
        self.max_epochs = max_epochs
        self.min_lr = min_lr if min_lr is not None else 0.
    '''step'''
    def step(self):
        raise NotImplementedError('not to be implemented')
    '''zerograd'''
    def zerograd(self):
        self.optimizer.zero_grad()
    '''updatelr'''
    def updatelr(self):
        raise NotImplementedError('not to be implemented')
    '''state'''
    def state(self):
        state_dict = {
            'cur_epoch': self.cur_epoch,
        }
        return state_dict
    '''load'''
    def load(self, state_dict):
        self.cur_iter = state_dict['iters_per_epoch'] * state_dict['cur_epoch']
        self.cur_epoch = state_dict['cur_epoch'] + 1
    '''clipgradients'''
    def clipgradients(self, params, max_norm=35, norm_type=2):
        params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            clip_grad.clip_grad_norm_(params, max_norm=max_norm, norm_type=norm_type)