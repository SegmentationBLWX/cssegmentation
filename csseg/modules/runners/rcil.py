'''
Function:
    Implementation of "Representation Compensation Networks for Continual Semantic Segmentation"
Author:
    Zhenchao Jin
'''
import copy
import math
import torch
import functools
import torch.nn.functional as F
import torch.distributed as dist
from apex import amp
from .base import BaseRunner


'''RCILRunner'''
class RCILRunner(BaseRunner):
    def __init__(self, mode, cmd_args, runner_cfg):
        super(RCILRunner, self).__init__(
            mode=mode, cmd_args=cmd_args, runner_cfg=runner_cfg
        )
    '''convertsegmentor'''
    def convertsegmentor(self):
        # merge
        def merge(conv2d, bn2d, conv_bias=None):
            if conv_bias is not None: conv_bias = conv_bias.clone().to(conv2d.weight.device)
            k = conv2d.weight.clone()
            running_mean, running_var, eps = bn2d.running_mean, bn2d.running_var, bn2d.eps
            gamma, beta = bn2d.weight.abs() + eps, bn2d.bias
            gamma, beta = gamma / 2., beta / 2.
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            if conv_bias is not None:
                return k * t, beta - running_mean * gamma / std + t.view(-1) * conv_bias.view(-1)
            else:
                return k * t, beta - running_mean * gamma / std
        # mergex
        def mergex(conv2d, bn2d, index, conv_bias=None, feats_channels=256):
            if conv_bias is not None: conv_bias = conv_bias.clone().to(conv2d.weight.device)
            k = conv2d.weight.clone()
            running_mean = bn2d.running_mean[index * feats_channels: (1 + index) * feats_channels]
            running_var = bn2d.running_var[index * feats_channels: (1 + index) * feats_channels]
            eps = bn2d.eps
            gamma = bn2d.weight.abs()[index * feats_channels: (1 + index) * feats_channels] + eps
            beta = bn2d.bias[index * feats_channels: (1 + index) * feats_channels]
            gamma, beta = gamma / 2., beta / 2.
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            if conv_bias is not None:
                return k * t, beta - running_mean * gamma / std + t.view(-1) * conv_bias.view(-1)
            else:
                return k * t, beta - running_mean * gamma / std
        # iter to convert
        for name, module in self.segmentor.named_modules():
            if hasattr(module, 'conv2') and hasattr(module, 'bn2') and hasattr(module, 'conv2_branch2') and hasattr(module, 'bn2_branch2'):
                module.conv2.bias = nn.Parameter(torch.zeros(module.conv2.weight.shape[0]).to(module.conv2.weight.device))
            elif hasattr(module, 'parallel_convs_branch1'):
                for idx in range(len(module.parallel_convs_branch1)):
                    module.parallel_convs_branch1[idx].bias = nn.Parameter(torch.zeros(module.parallel_convs_branch1[idx].weight.shape[0]).to(module.parallel_convs_branch1[idx].weight.device))
        for name, module in self.segmentor.named_modules():
            if hasattr(module, 'conv2') and hasattr(module, 'bn2') and hasattr(module, 'conv2_branch2') and hasattr(module, 'bn2_branch2'):
                k1, b1 = merge(module.conv2, module.bn2, module.conv2.bias.data)
                k2, b2 = merge(module.conv2_branch2, module.bn2_branch2, None)
                k, b = k1 + k2, b1 + b2
                module.conv2.weight.data[:, :, :, :] = k[:, :, :, :]
                module.conv2.bias = nn.Parameter(b)
                module.bn2.bias.data[:] = torch.zeros((module.bn2.weight.shape[0],))[:]
                module.bn2.running_var.data[:] = torch.ones((module.bn2.weight.shape[0],))[:]
                module.bn2.eps = 0
                module.bn2.weight.data[:] = torch.ones((module.bn2.weight.shape[0],))[:]
                module.bn2.running_mean.data[:] = torch.zeros((module.bn2.weight.shape[0],))[:]
                module.bn2.eval()
                module.conv2.eval()
                for param in module.bn2.parameters():
                    param.requires_grad = False
                for param in module.conv2.parameters():
                    param.requires_grad = False
            elif hasattr(module, 'parallel_convs_branch1'):
                for idx in range(len(module.parallel_convs_branch1)):
                    k1, b1 = mergex(module.parallel_convs_branch1[idx], module.parallel_bn_branch1[0], idx, module.parallel_convs_branch1[idx].bias.data)
                    k2, b2 = mergex(module.parallel_convs_branch2[idx], module.parallel_bn_branch2[0], idx, None)
                    k, b = k1 + k2, b1 + b2
                    module.parallel_convs_branch1[idx].weight.data[:, :, :, :] = k[:, :, :, :]
                    module.parallel_convs_branch1[idx].bias = nn.Parameter(b)
                    module.parallel_convs_branch1[idx].eval()
                    for param in module.parallel_convs_branch1[idx].parameters():
                        param.requires_grad = False
                module.parallel_bn_branch1[0].bias.data[:] = torch.zeros((module.parallel_bn_branch1[0].weight.shape[0],))[:]
                module.parallel_bn_branch1[0].running_var.data[:] = torch.ones((module.parallel_bn_branch1[0].weight.shape[0],))[:]
                module.parallel_bn_branch1[0].eps = 0
                module.parallel_bn_branch1[0].weight.data[:] = torch.ones((module.parallel_bn_branch1[0].weight.shape[0],))[:]
                module.parallel_bn_branch1[0].running_mean.data[:] = torch.zeros((module.parallel_bn_branch1[0].weight.shape[0],))[:]
                module.parallel_bn_branch1.eval()
                for param in module.parallel_bn_branch1.parameters():
                    param.requires_grad = False  
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
                    history_outputs = self.history_segmentor(images)
                    history_distillation_feats = history_outputs['distillation_feats']
                    history_distillation_feats.append(history_outputs['seg_logits'])
            # --forward to segmentor
            outputs = self.segmentor(images)
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
            pod_total_loss, pod_losses_log_dict = 0, {}
            if self.history_segmentor is not None:
                distillation_feats = outputs['distillation_feats']
                distillation_feats.append(outputs['seg_logits'])
                pod_total_loss, pod_losses_log_dict = self.featuresdistillation(
                    history_distillation_feats=history_distillation_feats, 
                    distillation_feats=distillation_feats,
                    num_known_classes_list=self.runner_cfg['segmentor_cfg']['num_known_classes_list'],
                    dataset_type=self.runner_cfg['dataset_cfg']['type'],
                    **losses_cfgs['distillation']
                )
            # --merge two losses
            loss_total = pod_total_loss + seg_total_loss
            # --perform back propagation
            with amp.scale_loss(loss_total, self.optimizer) as scaled_loss_total:
                scaled_loss_total.backward()
            self.scheduler.step()
            # --set zero gradient
            self.scheduler.zerograd()
            # --logging training loss info
            seg_losses_log_dict.update(pod_losses_log_dict)
            seg_losses_log_dict.pop('loss_total')
            seg_losses_log_dict['loss_total'] = loss_total.item()
            losses_log_dict = self.loggingtraininginfo(seg_losses_log_dict, losses_log_dict, init_losses_log_dict)
    '''featuresdistillation'''
    def featuresdistillation(self, history_distillation_feats, distillation_feats, num_known_classes_list=None, dataset_type='VOCDataset', scale_factor=1.0, spp_scales=[4, 8, 12, 16, 20, 24]):
        pod_total_loss = self.featuresdistillationchannel(history_distillation_feats, distillation_feats, num_known_classes_list, dataset_type) + \
            self.featuresdistillationspatial(history_distillation_feats, distillation_feats, num_known_classes_list, dataset_type, spp_scales)
        pod_total_loss = pod_total_loss * scale_factor
        value = pod_total_loss.data.clone()
        dist.all_reduce(value.div_(dist.get_world_size()))
        pod_losses_log_dict = {'loss_pod': value.item()}
        return pod_total_loss, pod_losses_log_dict
    '''featuresdistillationchannel'''
    @staticmethod
    def featuresdistillationchannel(history_distillation_feats, distillation_feats, num_known_classes_list=None, dataset_type='VOCDataset'):
        # assert and initialize
        assert len(history_distillation_feats) == len(distillation_feats)
        device = history_distillation_feats[0].device
        loss = torch.tensor(0.).to(device)
        num_known_classes = functools.reduce(lambda a, b: a + b, num_known_classes_list)
        num_curtask_classes = num_known_classes_list[-1]
        num_history_known_classes = num_known_classes - num_curtask_classes
        distillation_feats = distillation_feats[:-1]
        history_distillation_feats = history_distillation_feats[:-1]
        # start to iter
        for idx, (history_distillation, distillation) in enumerate(zip(history_distillation_feats, distillation_feats)):
            if history_distillation.shape[1] != distillation.shape[1]:
                distillation_tmp = torch.zeros_like(history_distillation).to(history_distillation.dtype).to(device)
                distillation_tmp[:, 0] = distillation[:, 0] + distillation[:, num_history_known_classes:].sum(dim=1)
                distillation_tmp[:, 1:] = distillation[:, 1:num_history_known_classes]
                distillation = distillation_tmp
            history_distillation, distillation = history_distillation ** 2, distillation ** 2
            history_distillation_p = F.avg_pool2d(history_distillation.permute(0, 2, 1, 3), (3, 1), stride=1, padding=(1, 0))
            distillation_p = F.avg_pool2d(distillation.permute(0, 2, 1, 3), (3, 1), stride=1, padding=(1, 0))
            layer_loss = torch.frobenius_norm((history_distillation_p - distillation_p).view(history_distillation.shape[0], -1), dim=-1).mean()
            if idx == len(history_distillation_feats) - 1:
                if dataset_type == 'ADE20kDataset':
                    pckd_factor = 5e-7
                elif dataset_type == 'VOCDataset':
                    pckd_factor = 0.0005
            else:
                if dataset_type == 'ADE20kDataset':
                    pckd_factor = 5e-6
                elif dataset_type == 'VOCDataset':
                    pckd_factor = 0.01
            loss = loss + layer_loss.mean() * math.sqrt(num_known_classes / num_curtask_classes) * pckd_factor
        # summarize and return
        loss = loss / len(history_distillation_feats)
        return loss
    '''featuresdistillationspatial'''
    @staticmethod
    def featuresdistillationspatial(history_distillation_feats, distillation_feats, num_known_classes_list=None, dataset_type='VOCDataset', spp_scales=[4, 8, 12, 16, 20, 24]):
        # assert and initialize
        assert len(history_distillation_feats) == len(distillation_feats)
        device = history_distillation_feats[0].device
        loss = torch.tensor(0.).to(device)
        num_known_classes = functools.reduce(lambda a, b: a + b, num_known_classes_list)
        num_curtask_classes = num_known_classes_list[-1]
        num_history_known_classes = num_known_classes - num_curtask_classes
        # start to iter
        for idx, (history_distillation, distillation) in enumerate(zip(history_distillation_feats, distillation_feats)):
            if history_distillation.shape[1] != distillation.shape[1]:
                distillation_tmp = torch.zeros_like(history_distillation).to(history_distillation.dtype).to(device)
                distillation_tmp[:, 0] = distillation[:, 0] + distillation[:, num_history_known_classes:].sum(dim=1)
                distillation_tmp[:, 1:] = distillation[:, 1:num_history_known_classes]
                distillation = distillation_tmp
            history_distillation, distillation = history_distillation ** 2, distillation ** 2
            layer_loss = torch.tensor(0.).to(device)
            for spp_scale in spp_scales:
                history_distillation_affinity = F.avg_pool2d(history_distillation, (spp_scale, spp_scale), stride=1, padding=spp_scale//2)
                distillation_affinity = F.avg_pool2d(distillation, (spp_scale, spp_scale), stride=1, padding=spp_scale//2)
                layer_loss = layer_loss + torch.frobenius_norm((history_distillation_affinity - distillation_affinity).view(history_distillation.shape[0], -1), dim=-1).mean()
            layer_loss = layer_loss / len(spp_scales)
            if idx == len(history_distillation_feats) - 1:
                if dataset_type == 'ADE20kDataset':
                    pckd_factor = 5e-7
                elif dataset_type == "voc":
                    pckd_factor = 0.0005
            else:
                if dataset_type == 'ADE20kDataset':
                    pckd_factor = 5e-6
                elif dataset_type == 'VOCDataset':
                    pckd_factor = 0.01
            loss = loss + layer_loss.mean() * math.sqrt(num_known_classes / num_curtask_classes) * pckd_factor
        # summarize and return
        loss = loss / len(history_distillation_feats)
        return loss