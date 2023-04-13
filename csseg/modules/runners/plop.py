'''
Function:
    Implementation of "PLOP: Learning without Forgetting for Continual Semantic Segmentation"
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
from tqdm import tqdm
from .base import BaseRunner


'''PLOPRunner'''
class PLOPRunner(BaseRunner):
    def __init__(self, mode, cmd_args, runner_cfg):
        super(PLOPRunner, self).__init__(
            mode=mode, cmd_args=cmd_args, runner_cfg=runner_cfg
        )
    '''preparefortrain'''
    def preparefortrain(self):
        if self.history_segmentor is not None:
            self.thresholds, self.max_entropy = self.findmedianforpseudolabeling()
    '''train'''
    def train(self, cur_epoch):
        # logging start task info
        if self.cmd_args.local_rank == 0:
            self.logger_handle.info(f'Start to train {self.runner_cfg["algorithm"]} at Task {self.runner_cfg["task_id"]}, Epoch {cur_epoch}')
        # initialize
        losses_cfgs = self.runner_cfg['LOSSES_CFGS']
        init_losses_log_dict = {
            'algorithm': self.runner_cfg['algorithm'], 'task_id': self.runner_cfg['task_id'],
            'epoch': self.scheduler.cur_epoch, 'iteration': self.scheduler.cur_iter, 'lr': self.scheduler.cur_lr
        }
        losses_log_dict = copy.deepcopy(init_losses_log_dict)
        self.segmentor.train()
        self.train_loader.sampler.set_epoch(cur_epoch)
        if self.history_segmentor is not None:
            thresholds, max_entropy = self.thresholds, self.max_entropy
        # start to iter
        for batch_idx, data_meta in enumerate(self.train_loader):
            # --fetch data
            images = data_meta['image'].to(self.device, dtype=torch.float32)
            targets = data_meta['target'].to(self.device, dtype=torch.long)
            seg_targets = targets.clone()
            # --pseudo labeling
            classifier_adaptive_factor = 1.0
            if self.history_segmentor is not None:
                num_history_known_classes = functools.reduce(lambda a, b: a + b, self.runner_cfg['SEGMENTOR_CFG']['num_known_classes_list'][:-1])
                history_outputs = self.history_segmentor(images)
                background_mask = (targets < num_history_known_classes)
                history_seg_logits = F.interpolate(history_outputs['seg_logits'], size=images.shape[2:], mode="bilinear", align_corners=self.segmentor.module.align_corners)
                seg_probs = torch.softmax(history_seg_logits, dim=1)
                max_seg_probs, pseudo_labels = seg_probs.max(dim=1)
                valid_pseudo_mask = (self.entropy(seg_probs) / max_entropy) < thresholds[pseudo_labels]
                seg_targets[~valid_pseudo_mask & background_mask] = 255
                seg_targets[valid_pseudo_mask & background_mask] = pseudo_labels[valid_pseudo_mask & background_mask]
                classifier_adaptive_factor = (valid_pseudo_mask & background_mask).float().sum(dim=(1, 2)) / (background_mask.float().sum(dim=(1, 2)) + self.eps)
                classifier_adaptive_factor = classifier_adaptive_factor[:, None, None]
            # --set zero gradient
            self.scheduler.zerograd()
            # --forward to segmentor
            outputs = self.segmentor(images)
            # --calculate segmentation losses
            seg_losses_cfgs = copy.deepcopy(losses_cfgs['segmentation'])
            for _, seg_losses_cfg in seg_losses_cfgs.items():
                for loss_type, loss_cfg in seg_losses_cfg.items():
                    loss_cfg.update({'scale_factor': classifier_adaptive_factor, 'reduction': 'none'})
            seg_total_loss, seg_losses_log_dict = self.segmentor.module.calculateseglosses(
                seg_logits=outputs['seg_logits'], 
                seg_targets=seg_targets, 
                losses_cfgs=seg_losses_cfgs,
            )
            # --calculate distillatio losses
            pod_total_loss, pod_losses_log_dict = 0, {}
            if self.history_segmentor is not None:
                attentions, history_attentions = outputs['attentions'], history_outputs['attentions']
                attentions.append(outputs['seg_logits'])
                history_attentions.append(outputs['seg_logits'])
                pod_total_loss, pod_losses_log_dict = self.featuresdistillation(
                    history_attentions=history_attentions, 
                    attentions=attentions,
                    num_known_classes_list=self.runner_cfg['SEGMENTOR_CFG']['num_known_classes_list'],
                    **losses_cfgs['distillation']
                )
            # --merge two losses
            loss_total = pod_total_loss + seg_total_loss
            # --perform back propagation
            with amp.scale_loss(loss_total, self.optimizer) as scaled_loss_total:
                scaled_loss_total.backward()
            self.scheduler.step()
            # --logging training loss info
            seg_losses_log_dict.update(pod_losses_log_dict)
            seg_losses_log_dict.pop('loss_total')
            for key, value in seg_losses_log_dict.items():
                if key in losses_log_dict:
                    losses_log_dict[key].append(value)
                else:
                    losses_log_dict[key] = [value]
            if 'loss_total' in losses_log_dict:
                losses_log_dict['loss_total'].append(loss_total.item())
            else:
                losses_log_dict['loss_total'] = [loss_total.item()]
            losses_log_dict.update({
                'epoch': self.scheduler.cur_epoch, 'iteration': self.scheduler.cur_iter, 'lr': self.scheduler.cur_lr
            })
            if (self.scheduler.cur_iter % self.log_interval_iterations == 0) and (self.cmd_args.local_rank == 0):
                for key, value in losses_log_dict.copy().items():
                    if isinstance(value, list):
                        losses_log_dict[key] = sum(value) / len(value)
                self.logger_handle.info(losses_log_dict)
                losses_log_dict = copy.deepcopy(init_losses_log_dict)
    '''findmedianforpseudolabeling'''
    def findmedianforpseudolabeling(self):
        # initialize
        num_known_classes = functools.reduce(lambda a, b: a + b, self.runner_cfg['SEGMENTOR_CFG']['num_known_classes_list'])
        max_value = torch.log(torch.tensor(num_known_classes).float().to(self.device))
        num_bins = 100
        histograms = torch.zeros(num_known_classes, num_bins).long().to(self.device)
        # start to iter
        train_loader = self.train_loader
        if self.cmd_args.local_rank == 0:
            train_loader = tqdm(train_loader)
            train_loader.set_description('Find Pseudo Labeling Median')
        for batch_idx, data_meta in enumerate(train_loader):
            images = data_meta['image'].to(self.device, dtype=torch.float32)
            targets = data_meta['target'].to(self.device, dtype=torch.long)
            seg_logits = self.history_segmentor(images)['seg_logits']
            seg_logits = F.interpolate(seg_logits, size=images.shape[2:], mode="bilinear", align_corners=self.segmentor.module.align_corners)
            background_mask = (targets == 0)
            seg_probs = torch.softmax(seg_logits, dim=1)
            max_seg_probs, pseudo_labels = seg_probs.max(dim=1)
            values_to_bins = self.entropy(seg_probs)[background_mask].view(-1) / max_value
            x_coords = pseudo_labels[background_mask].view(-1)
            y_coords = torch.clamp((values_to_bins * num_bins).long(), max=num_bins - 1)
            histograms.index_put_((x_coords, y_coords), torch.LongTensor([1]).expand_as(x_coords).to(histograms.device), accumulate=True)
        # calculate thresholds
        thresholds = torch.zeros(num_known_classes, dtype=torch.float32).to(self.device)
        for cls_id in range(num_known_classes):
            total = histograms[cls_id].sum()
            if total <= 0.: continue
            half = total / 2
            running_sum = 0.
            for lower_border in range(num_bins):
                lower_border = lower_border / num_bins
                bin_index = int(lower_border * num_bins)
                if half >= running_sum and half <= (running_sum + histograms[cls_id, bin_index]):
                    break
                running_sum += lower_border * num_bins
            median = lower_border + ((half - running_sum) / histograms[cls_id, bin_index].sum()) * (1 / num_bins)
            thresholds[cls_id] = median
        # set pseudolabeling_minimal_threshold
        pseudolabeling_minimal_threshold = self.runner_cfg['pseudolabeling_minimal_threshold']
        for cls_id in range(len(thresholds)):
            thresholds[cls_id] = max(thresholds[cls_id], pseudolabeling_minimal_threshold)
        # return
        return thresholds.to(self.device), max_value
    '''entropy'''
    @staticmethod
    def entropy(probabilities, eps=1e-8):
        factor = 1 / math.log(probabilities.shape[1] + eps)
        return -factor * torch.mean(probabilities * torch.log(probabilities + eps), dim=1)
    '''featuresdistillation'''
    @staticmethod
    def featuresdistillation(history_attentions, attentions, pod_factor=0.01, pod_factor_last_scale=0.0005, spp_scales=[1, 2, 4], num_known_classes_list=None):
        # assert and initialize
        assert len(history_attentions) == len(attentions)
        device = history_attentions[0].device
        loss = torch.tensor(0.).to(device)
        num_known_classes = functools.reduce(lambda a, b: a + b, num_known_classes_list)
        num_curtask_classes = num_known_classes_list[-1]
        num_history_known_classes = num_known_classes - num_curtask_classes
        # start to iter
        for idx, (history_attention, attention) in enumerate(zip(history_attentions, attentions)):
            if idx == len(history_attentions) - 1:
                pod_factor = pod_factor_last_scale if pod_factor_last_scale is not None else pod_factor
            if history_attention.shape[1] != attention.shape[1]:
                tmp = torch.zeros_like(history_attention).to(history_attention.dtype).to(history_attention.device)
                tmp[:, 0] = attention[:, 0] + attention[:, num_history_known_classes:].sum(dim=1)
                tmp[:, 1:] = attention[:, 1:num_history_known_classes]
                attention = tmp
            history_attention = torch.pow(history_attention, 2)
            history_attention = PLOPRunner.localpod(history_attention, spp_scales)
            attention = torch.pow(attention, 2)
            attention = PLOPRunner.localpod(attention, spp_scales)
            if isinstance(history_attention, list):
                layer_loss = torch.tensor([torch.frobenius_norm(h_a - n_a, dim=-1) for h_a, n_a in zip(history_attention, attention)]).to(device)
            else:
                layer_loss = torch.frobenius_norm(history_attention - attention, dim=-1)
            layer_loss = layer_loss.mean()
            layer_loss = pod_factor * layer_loss
            layer_loss = layer_loss * math.sqrt(num_known_classes / num_curtask_classes)
            loss += layer_loss
        pod_total_loss = loss / len(history_attentions)
        value = pod_total_loss.data.clone()
        dist.all_reduce(value.div_(dist.get_world_size()))
        pod_losses_log_dict = {'loss_pod': value.item()}
        return pod_total_loss, pod_losses_log_dict
    '''localpod'''
    @staticmethod
    def localpod(x, spp_scales=[1, 2, 4]):
        batch_size, num_channels, height, width = x.shape
        embeddings = []
        for scale_idx, scale in enumerate(spp_scales):
            pod_size = width // scale
            for i in range(scale):
                for j in range(scale):
                    tensor = x[..., i * pod_size: (i + 1) * pod_size, j * pod_size: (j + 1) * pod_size]
                    horizontal_pool = tensor.mean(dim=3).view(batch_size, -1)
                    vertical_pool = tensor.mean(dim=2).view(batch_size, -1)
                    embeddings.append(horizontal_pool)
                    embeddings.append(vertical_pool)
        return torch.cat(embeddings, dim=1)