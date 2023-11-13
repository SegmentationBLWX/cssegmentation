'''
Function:
    Implementation of "Modeling the Background for Incremental Learning in Semantic Segmentation"
Author:
    Zhenchao Jin
'''
import copy
import torch
import functools
import torch.nn.functional as F
import torch.distributed as dist
from .base import BaseRunner


'''MIBRunner'''
class MIBRunner(BaseRunner):
    def __init__(self, mode, cmd_args, runner_cfg):
        super(MIBRunner, self).__init__(
            mode=mode, cmd_args=cmd_args, runner_cfg=runner_cfg
        )
    '''call'''
    def __call__(self, images, seg_targets):
        # initialize
        losses_cfgs = copy.deepcopy(self.losses_cfgs)
        # feed to history_segmentor
        if self.history_segmentor is not None:
            with torch.no_grad():
                history_outputs = self.history_segmentor(images)
        # feed to segmentor
        outputs = self.segmentor(images)
        # calculate segmentation losses
        seg_losses_cfgs = copy.deepcopy(losses_cfgs['segmentation_cl']) if self.history_segmentor is not None else copy.deepcopy(losses_cfgs['segmentation_init'])
        if self.history_segmentor is not None:
            num_history_known_classes = functools.reduce(lambda a, b: a + b, self.runner_cfg['segmentor_cfg']['num_known_classes_list'][:-1])
            for _, seg_losses_cfg in seg_losses_cfgs.items():
                for loss_type, loss_cfg in seg_losses_cfg.items():
                    loss_cfg.update({'num_history_known_classes': num_history_known_classes})
        seg_total_loss, seg_losses_log_dict = self.segmentor.module.calculateseglosses(
            seg_logits=outputs['seg_logits'], seg_targets=seg_targets, losses_cfgs=seg_losses_cfgs,
        )
        # calculate distillation losses
        kd_total_loss, kd_losses_log_dict = 0, {}
        if self.history_segmentor is not None:
            kd_total_loss, kd_losses_log_dict = self.featuresdistillation(
                history_distillation_feats=F.interpolate(history_outputs['seg_logits'], size=images.shape[2:], mode="bilinear", align_corners=self.segmentor.module.align_corners), 
                distillation_feats=F.interpolate(outputs['seg_logits'], size=images.shape[2:], mode="bilinear", align_corners=self.segmentor.module.align_corners),
                **losses_cfgs['distillation']
            )
        # deal with losses
        loss_total = kd_total_loss + seg_total_loss
        seg_losses_log_dict.update(kd_losses_log_dict)
        seg_losses_log_dict.pop('loss_total')
        seg_losses_log_dict['loss_total'] = loss_total.item()
        # return
        return loss_total, seg_losses_log_dict
    '''featuresdistillation'''
    @staticmethod
    def featuresdistillation(history_distillation_feats, distillation_feats, reduction='mean', alpha=1., scale_factor=10):
        new_cl = distillation_feats.shape[1] - history_distillation_feats.shape[1]
        history_distillation_feats = history_distillation_feats * alpha
        new_bkg_idx = torch.tensor([0] + [x for x in range(history_distillation_feats.shape[1], distillation_feats.shape[1])]).to(distillation_feats.device)
        den = torch.logsumexp(distillation_feats, dim=1)
        outputs_no_bgk = distillation_feats[:, 1:-new_cl] - den.unsqueeze(dim=1)
        outputs_bkg = torch.logsumexp(torch.index_select(distillation_feats, index=new_bkg_idx, dim=1), dim=1) - den
        labels = torch.softmax(history_distillation_feats, dim=1)
        loss = (labels[:, 0] * outputs_bkg + (labels[:, 1:] * outputs_no_bgk).sum(dim=1)) / history_distillation_feats.shape[1]
        if reduction == 'mean': 
            loss = -torch.mean(loss)
        elif reduction == 'sum':
            loss = -torch.sum(loss)
        else:
            loss = -loss
        loss = loss * scale_factor
        value = loss.data.clone()
        dist.all_reduce(value.div_(dist.get_world_size()))
        kd_losses_log_dict = {'loss_kd': value.item()}
        return loss, kd_losses_log_dict