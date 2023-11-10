'''
Function:
    Implementation of "Incremental learning techniques for semantic segmentation"
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn.functional as F
import torch.distributed as dist
from .base import BaseRunner
from ..models import BuildLoss
from torch.cuda.amp import autocast


'''ILTRunner'''
class ILTRunner(BaseRunner):
    def __init__(self, mode, cmd_args, runner_cfg):
        super(ILTRunner, self).__init__(
            mode=mode, cmd_args=cmd_args, runner_cfg=runner_cfg
        )
    '''train'''
    def train(self, cur_epoch):
        # initialize
        losses_cfgs = copy.deepcopy(self.losses_cfgs)
        init_losses_log_dict = {
            'algorithm': self.runner_cfg['algorithm'], 'work_tag': self.cmd_args.cfgfilepath.split('/')[-1][:-3], 'task_name': self.runner_cfg['task_name'], 'task_id': self.runner_cfg['task_id'], 
            'encoder': self.runner_cfg['segmentor_cfg']['encoder_cfg']['type'], 'decoder': self.runner_cfg['segmentor_cfg']['decoder_cfg']['type'],
            'cur_epoch': self.scheduler.cur_epoch, 'max_epochs': self.scheduler.max_epochs, 'cur_iter': self.scheduler.cur_iter, 'max_iters': self.scheduler.max_iters,
            'lr': self.scheduler.lr,
        }
        losses_log_dict = copy.deepcopy(init_losses_log_dict)
        self.segmentor.train()
        self.train_loader.sampler.set_epoch(cur_epoch)
        # start to iter
        for batch_idx, data_meta in enumerate(self.train_loader):
            # --fetch data
            images = data_meta['image'].to(self.device, dtype=torch.float32)
            seg_targets = data_meta['seg_target'].to(self.device, dtype=torch.long)
            # --set zero gradient
            self.scheduler.zerograd()
            # --autocast
            with autocast(**self.runner_cfg['fp16_cfg']['autocast']):
                # ----feed to history_segmentor
                if self.history_segmentor is not None:
                    with torch.no_grad():
                        history_outputs = self.history_segmentor(images)
                # ----forward to segmentor
                outputs = self.segmentor(images)
                # ----calculate segmentation losses
                seg_losses_cfgs = copy.deepcopy(losses_cfgs['segmentation_cl']) if self.history_segmentor is not None else copy.deepcopy(losses_cfgs['segmentation_init'])
                seg_total_loss, seg_losses_log_dict = self.segmentor.module.calculateseglosses(
                    seg_logits=outputs['seg_logits'], 
                    seg_targets=seg_targets, 
                    losses_cfgs=seg_losses_cfgs,
                )
                # ----calculate distillation losses
                kd_total_loss, kd_losses_log_dict = 0, {}
                if self.history_segmentor is not None:
                    kd_loss_logits, kd_losses_log_dict = self.featuresdistillation(
                        history_distillation_feats=F.interpolate(history_outputs['seg_logits'], size=images.shape[2:], mode="bilinear", align_corners=self.segmentor.module.align_corners), 
                        distillation_feats=F.interpolate(outputs['seg_logits'], size=images.shape[2:], mode="bilinear", align_corners=self.segmentor.module.align_corners),
                        **losses_cfgs['distillation_logits']
                    )
                    kd_loss_feats = BuildLoss(losses_cfgs['distillation_features'])(prediction=outputs['distillation_feats'], target=history_outputs['distillation_feats'])
                    value = kd_loss_feats.data.clone()
                    dist.all_reduce(value.div_(dist.get_world_size()))
                    kd_losses_log_dict['kd_loss_feats'] = value.item()
                    kd_total_loss = kd_loss_logits + kd_loss_feats
                # ----merge two losses
                loss_total = kd_total_loss + seg_total_loss
            # --perform back propagation
            self.grad_scaler.scale(loss_total).backward()
            self.scheduler.step(self.grad_scaler)
            # --logging training loss info
            seg_losses_log_dict.update(kd_losses_log_dict)
            seg_losses_log_dict.pop('loss_total')
            seg_losses_log_dict['loss_total'] = loss_total.item()
            losses_log_dict = self.loggingtraininginfo(seg_losses_log_dict, losses_log_dict, init_losses_log_dict)
    '''featuresdistillation'''
    @staticmethod
    def featuresdistillation(history_distillation_feats, distillation_feats, reduction='mean', alpha=1., scale_factor=100):
        distillation_feats = distillation_feats.narrow(1, 0, history_distillation_feats.shape[1])
        outputs = torch.log_softmax(distillation_feats, dim=1)
        labels = torch.softmax(history_distillation_feats * alpha, dim=1)
        loss = (outputs * labels).mean(dim=1)
        if reduction == 'mean': 
            loss = -torch.mean(loss)
        elif reduction == 'sum':
            loss = -torch.sum(loss)
        else:
            loss = -loss
        loss = loss * scale_factor
        value = loss.data.clone()
        dist.all_reduce(value.div_(dist.get_world_size()))
        kd_losses_log_dict = {'kd_loss_logits': value.item()}
        return loss, kd_losses_log_dict