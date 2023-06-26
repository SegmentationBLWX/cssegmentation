'''
Function:
    Implementation of "Uncertainty-aware Contrastive Distillation for Incremental Semantic Segmentation"
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


'''UCDRunner'''
class UCDRunner(BaseRunner):
    def __init__(self, mode, cmd_args, runner_cfg):
        super(MIBRunner, self).__init__(
            mode=mode, cmd_args=cmd_args, runner_cfg=runner_cfg
        )
    '''train'''
    def train(self, cur_epoch):
        # initialize
        losses_cfgs = copy.deepcopy(self.losses_cfgs)
        init_losses_log_dict = {
            'algorithm': self.runner_cfg['algorithm'], 'task_id': self.runner_cfg['task_id'],
            'epoch': self.scheduler.cur_epoch, 'iteration': self.scheduler.cur_iter, 'lr': self.scheduler.cur_lr
        }
        losses_log_dict = copy.deepcopy(init_losses_log_dict)
        self.segmentor.train()
        self.train_loader.sampler.set_epoch(cur_epoch)
        # start to iter
        for batch_idx, data_meta in enumerate(self.train_loader):
            # --fetch data
            images = data_meta['image'].to(self.device, dtype=torch.float32)
            seg_targets = data_meta['seg_target'].to(self.device, dtype=torch.long)
    '''featuresdistillation'''
    @staticmethod
    def featuresdistillation(anchor_features, contrast_features, anchor_labels, contrast_labels, P=None):
        device = anchor_features.device
        anchor_labels = anchor_labels.view(-1, 1)
        contrast_labels = contrast_labels.view(-1, 1)
        batch_size = anchor_features.shape[0]
        R = torch.eq(anchor_labels, contrast_labels.T).float().requires_grad_(False).to(device)
        mask_p = R.clone().requires_grad_(False)
        mask_p[:, :batch_size] -= torch.eye(batch_size).to(device)
        mask_p = mask_p.detach()
        mask_n = 1 - R
        mask_n = mask_n.detach()
        anchor_dot_contrast = torch.div(torch.mm(anchor_features, contrast_features.T), self.temperature)
        neg_contrast = (torch.exp(anchor_dot_contrast) * mask_n).sum(dim=1, keepdim=True)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        anchor_dot_contrast = anchor_dot_contrast - logits_max.detach()
        if P is None:
            pos_contrast = torch.log(torch.exp(anchor_dot_contrast)) * mask_p - torch.log(torch.exp(anchor_dot_contrast) + neg_contrast) * mask_p
        else:
            pos_contrast = torch.log(torch.exp(anchor_dot_contrast)) * mask_p * P - torch.log(torch.exp(anchor_dot_contrast) + neg_contrast) * mask_p * P
        num = mask_p.sum(dim=1)
        loss = -torch.div(pos_contrast.sum(dim=1)[num != 0], num[num != 0])
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        loss = loss * self.scale_factor
        return loss