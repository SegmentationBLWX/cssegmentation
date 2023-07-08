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
from .mib import MIBRunner


'''UCDMIBRunner'''
class UCDMIBRunner(MIBRunner):
    def __init__(self, mode, cmd_args, runner_cfg):
        super(UCDMIBRunner, self).__init__(
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
            # --feed to history_segmentor
            if self.history_segmentor is not None:
                with torch.no_grad():
                    history_outputs = self.history_segmentor(images, task_id=self.runner_cfg['task_id'])
            # --forward to segmentor
            outputs = self.segmentor(images, task_id=self.runner_cfg['task_id'])
            # --calculate segmentation losses
            seg_losses_cfgs = copy.deepcopy(losses_cfgs['segmentation_cl']) if self.history_segmentor is not None else copy.deepcopy(losses_cfgs['segmentation_init'])
            if self.history_segmentor is not None:
                num_history_known_classes = functools.reduce(lambda a, b: a + b, self.runner_cfg['segmentor_cfg']['num_known_classes_list'][:-1])
                for _, seg_losses_cfg in seg_losses_cfgs.items():
                    for loss_type, loss_cfg in seg_losses_cfg.items():
                        loss_cfg.update({'num_history_known_classes': num_history_known_classes})
            seg_total_loss, seg_losses_log_dict = self.segmentor.module.calculateseglosses(
                seg_logits=outputs['seg_logits'], 
                seg_targets=seg_targets, 
                losses_cfgs=seg_losses_cfgs,
            )
            # --calculate distillation losses
            kd_total_loss, kd_losses_log_dict = 0, {}
            if self.history_segmentor is not None:
                kd_total_loss, kd_losses_log_dict = self.featuresdistillation(
                    history_distillation_feats=F.interpolate(history_outputs['seg_logits'], size=images.shape[2:], mode="bilinear", align_corners=self.segmentor.module.align_corners), 
                    distillation_feats=F.interpolate(outputs['seg_logits'], size=images.shape[2:], mode="bilinear", align_corners=self.segmentor.module.align_corners),
                    **losses_cfgs['distillation']
                )
            # --calculate contrastive losses
            cl_total_loss, cl_losses_log_dict = 0, {}
            if self.history_segmentor is not None:
                anchor_features, contrast_features, anchor_labels, contrast_labels, P = self.preprocessforcontrastivelearning(
                    outputs['decoder_outputs'], seg_targets, history_outputs['seg_logits'], history_outputs['decoder_outputs']
                )
                cl_total_loss, cl_losses_log_dict = self.contrastivelearning(
                    anchor_features, contrast_features, anchor_labels, contrast_labels, P, **losses_cfgs['contrastive']
                )
            # --merge three losses
            loss_total = kd_total_loss + cl_total_loss + seg_total_loss
            # --perform back propagation
            with amp.scale_loss(loss_total, self.optimizer) as scaled_loss_total:
                scaled_loss_total.backward()
            self.scheduler.step()
            # --set zero gradient
            self.scheduler.zerograd()
            # --logging training loss info
            seg_losses_log_dict.update(kd_losses_log_dict)
            seg_losses_log_dict.update(cl_losses_log_dict)
            seg_losses_log_dict.pop('loss_total')
            seg_losses_log_dict['loss_total'] = loss_total.item()
            losses_log_dict = self.loggingtraininginfo(seg_losses_log_dict, losses_log_dict, init_losses_log_dict)
            # del outputs and perform empty_cache to save memory
            del outputs
            torch.cuda.empty_cache()
    '''contrastivelearning'''
    @staticmethod
    def contrastivelearning(anchor_features, contrast_features, anchor_labels, contrast_labels, P=None, temperature=0.07, scale_factor=1.0, reduction='mean'):
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
        anchor_dot_contrast = torch.div(torch.mm(anchor_features, contrast_features.T), temperature)
        neg_contrast = (torch.exp(anchor_dot_contrast) * mask_n).sum(dim=1, keepdim=True)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        anchor_dot_contrast = anchor_dot_contrast - logits_max.detach()
        if P is None:
            pos_contrast = torch.log(torch.exp(anchor_dot_contrast)) * mask_p - torch.log(torch.exp(anchor_dot_contrast) + neg_contrast) * mask_p
        else:
            pos_contrast = torch.log(torch.exp(anchor_dot_contrast)) * mask_p * P - torch.log(torch.exp(anchor_dot_contrast) + neg_contrast) * mask_p * P
        num = mask_p.sum(dim=1)
        loss = -torch.div(pos_contrast.sum(dim=1)[num != 0], num[num != 0])
        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        loss = loss * scale_factor
        value = loss.data.clone()
        dist.all_reduce(value.div_(dist.get_world_size()))
        cl_losses_log_dict = {'loss_cl': value.item()}
        return loss, cl_losses_log_dict
    '''preprocessforcontrastivelearning'''
    @staticmethod
    def preprocessforcontrastivelearning(decoder_outputs, seg_targets, history_seg_logits, history_decoder_outputs, ignore_index=255):
        assert decoder_outputs.shape[2:] == history_decoder_outputs.shape[2:] and decoder_outputs.shape[2:] == history_seg_logits.shape[2:]
        # re-arrange
        batch_size, num_channels, h, w = decoder_outputs.size()
        seg_targets = F.interpolate(seg_targets.detach().unsqueeze(1), size=decoder_outputs.shape[-2:], mode='nearest').long()
        decoder_outputs = decoder_outputs.permute(0, 2, 3, 1).contiguous().reshape(batch_size, h * w, num_channels)
        history_decoder_outputs = history_decoder_outputs.detach().permute(0, 2, 3, 1).contiguous().reshape(batch_size, h * w, num_channels)
        # merge pesudo labels to seg_targets
        mask_current_classes = seg_targets.view(-1) > 0
        current_classes_minclsid = seg_targets.view(-1)[mask_current_classes].min()
        seg_targets_mergepseudolabels = seg_targets.squeeze(1)
        seg_targets_mergepseudolabels[seg_targets_mergepseudolabels == 0] = history_seg_logits.max(dim=1)[1].type_as(seg_targets_mergepseudolabels)[seg_targets_mergepseudolabels == 0]
        seg_targets_mergepseudolabels = seg_targets_mergepseudolabels.reshape(batch_size * h * w)
        # obtain anchor_labels and contrast_labels
        anchor_labels = seg_targets_mergepseudolabels[(seg_targets_mergepseudolabels > 0) & (seg_targets_mergepseudolabels != ignore_index)].clone()
        contrast_labels = torch.cat([anchor_labels, seg_targets_mergepseudolabels[(seg_targets_mergepseudolabels > 0) & (seg_targets_mergepseudolabels != ignore_index) & ~mask_current_classes]], dim=0)
        # obtain anchor_features and contrast_features
        anchor_features = F.normalize(decoder_outputs.reshape(batch_size * h * w, num_channels)[(seg_targets_mergepseudolabels > 0) & (seg_targets_mergepseudolabels != ignore_index)], dim=1)
        contrast_features = torch.cat([anchor_features, F.normalize(history_decoder_outputs.reshape(batch_size * h * w, num_channels)[(seg_targets_mergepseudolabels > 0) & (seg_targets_mergepseudolabels != ignore_index) & ~mask_current_classes], dim=1)], dim=0).detach()
        # make joint probability mask
        history_seg_probs = torch.softmax(history_seg_logits.permute(0, 2, 3, 1), dim=-1)
        history_seg_probs = history_seg_probs.reshape(batch_size * h * w, -1)
        history_seg_probs_anchor = history_seg_probs[(seg_targets_mergepseudolabels > 0) & (seg_targets_mergepseudolabels != ignore_index)]
        history_seg_probs_contrast = torch.cat(
            [history_seg_probs_anchor, history_seg_probs[(seg_targets_mergepseudolabels > 0) & (seg_targets_mergepseudolabels != ignore_index) & ~mask_current_classes]], dim=0
        )
        JM_p = torch.mm(history_seg_probs_anchor, history_seg_probs_contrast.T)
        # mask old classes on anchor_labels
        mask_anchor_labels = torch.zeros_like(anchor_labels).to(anchor_labels.dtype).to(anchor_labels.device)
        mask_anchor_labels[mask_anchor_labels >= current_classes_minclsid] = 1
        # mask old classes on contrast_labels
        mask_contrast_labels = torch.zeros_like(contrast_labels).to(contrast_labels.dtype).to(contrast_labels.device)
        mask_contrast_labels[mask_contrast_labels >= current_classes_minclsid] = 1
        # fix gt with gt cases
        M_gt = torch.mm(mask_anchor_labels.unsqueeze(dim=1), mask_contrast_labels.unsqueeze(dim=1).T)
        JM_p[M_gt == 1] = 1
        # return
        return anchor_features, contrast_features, anchor_labels, contrast_labels, JM_p.detach()