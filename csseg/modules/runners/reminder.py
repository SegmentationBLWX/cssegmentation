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
from .plop import PLOPRunner


'''REMINDERRunner'''
class REMINDERRunner(PLOPRunner):
    def __init__(self, mode, cmd_args, runner_cfg):
        super(REMINDERRunner, self).__init__(
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
        if self.history_segmentor is not None:
            thresholds, max_entropy = self.thresholds, self.max_entropy
        # start to iter
        for batch_idx, data_meta in enumerate(self.train_loader):
            # --fetch data
            images = data_meta['image'].to(self.device, dtype=torch.float32)
            seg_targets = data_meta['seg_target'].to(self.device, dtype=torch.long)
            seg_targets_mergepseudolabels = seg_targets.clone()
            # --pseudo labeling
            classifier_adaptive_factor = 1.0
            if self.history_segmentor is not None:
                num_history_known_classes = functools.reduce(lambda a, b: a + b, self.runner_cfg['segmentor_cfg']['num_known_classes_list'][:-1])
                with torch.no_grad():
                    history_outputs = self.history_segmentor(images)
                    history_distillation_feats = history_outputs['distillation_feats']
                    history_distillation_feats.append(history_outputs['seg_logits'])
                    history_seg_logits = F.interpolate(history_outputs['seg_logits'], size=images.shape[2:], mode="bilinear", align_corners=self.segmentor.module.align_corners)
                background_mask = (seg_targets < num_history_known_classes)
                history_seg_probs = torch.softmax(history_seg_logits, dim=1)
                max_history_seg_probs, pseudo_labels = history_seg_probs.max(dim=1)
                valid_pseudo_mask = (self.entropy(history_seg_probs) / max_entropy) < thresholds[pseudo_labels]
                seg_targets_mergepseudolabels[~valid_pseudo_mask & background_mask] = 255
                seg_targets_mergepseudolabels[valid_pseudo_mask & background_mask] = pseudo_labels[valid_pseudo_mask & background_mask]
                classifier_adaptive_factor = (valid_pseudo_mask & background_mask).float().sum(dim=(1, 2)) / (background_mask.float().sum(dim=(1, 2)) + self.eps)
                classifier_adaptive_factor = classifier_adaptive_factor[:, None, None]
            # --forward to segmentor
            outputs = self.segmentor(images)
            # --calculate segmentation losses
            seg_losses_cfgs = copy.deepcopy(losses_cfgs['segmentation_cl']) if self.history_segmentor is not None else copy.deepcopy(losses_cfgs['segmentation_init'])
            for _, seg_losses_cfg in seg_losses_cfgs.items():
                for loss_type, loss_cfg in seg_losses_cfg.items():
                    loss_cfg.update({'scale_factor': classifier_adaptive_factor, 'reduction': 'none'})
            seg_total_loss, seg_losses_log_dict = self.segmentor.module.calculateseglosses(
                seg_logits=outputs['seg_logits'], 
                seg_targets=seg_targets_mergepseudolabels, 
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
                    **losses_cfgs['distillation']
                )
            cse_total_loss, cse_losses_log_dict = 0, {}
            if self.history_segmentor is not None:
                self.cswfeaturesdistillation(
                    logits_source=F.interpolate(outputs['seg_logits'], size=images.shape[2:], mode="bilinear", align_corners=self.segmentor.module.align_corners),
                    logits_target=F.interpolate(history_outputs['seg_logits'], size=images.shape[2:], mode="bilinear", align_corners=self.segmentor.module.align_corners),
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
    '''cswfeaturesdistillation'''
    @staticmethod
    def cswfeaturesdistillation(logits_source, logits_target, seg_targets_mergepseudolabels, prototypes, temperature=3, delta=0.0, history_prototypes=None):
        batch_prototypes.detach()
        logits_target.detach()
        logits_source = logits_source.narrow(1, 0, logits_target.shape[1])
        assert not torch.isnan(batch_prototypes).any(), "NaN in prototype"
        assert logits_source[0].shape == logits_target[0].shape, 'the output dim of teacher and student differ'
        B, _, H, W = logits_source.shape
        logits_source = logits_source.permute(0, 2, 3, 1).contiguous().view(B*H*W, -1)[mask] # T * N_old, where T=BxHxW
        logits_target = logits_target.permute(0, 2, 3, 1).contiguous().view(B*H*W, -1)[mask] # T * N_old
        seg = seg.view(B*H*W)[mask] # T
        T = seg.size(0)
        proto_by_label = batch_prototypes[seg] # T * C
        r_map = pairwise_cosine_sim(proto_by_label, self.prototypes.to(proto_by_label.device)) # T * N_old
        r_map = F.softmax(r_map, dim=-1)
        r_map[r_map < (self.delta/r_map.size(1))] = 0.0

        logits_source = F.log_softmax(logits_source / self.T, dim=1)
        logits_target = F.softmax(logits_target / self.T, dim=1)
        logits_target = logits_target * r_map + 10 ** (-7)
        logits_target = torch.autograd.Variable(logits_target.data.cuda(), requires_grad=False)
        loss = self.T * self.T * torch.sum(-logits_target*logits_source)/T
        return loss
    def update_prototypes(self, train_loader):
        device = self.device
        with torch.no_grad():
            for cur_step, (images, labels) in enumerate(train_loader):
                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)
                _, features = self.model(images, ret_intermediate=True)
                exist_label = labels[labels != 255].unique()
                pre_logits = features['pre_logits']
                if 'cityscapes' in self.dataset:
                    cur_classes = list(range(self.nb_current_classes))
                else:
                    cur_classes = list(range(self.old_classes, self.nb_current_classes))
                batch_prototypes = compute_prototype(labels, pre_logits, cur_classes)

                self.proto_count[exist_label] += 1
                #self.proto_count = self.proto_count.to(device)
                #batch_size = images.size(0)
                self.current_prototypes[exist_label] = (1 / self.proto_count[exist_label]).unsqueeze(1) * (
                        (self.proto_count[exist_label] - 1).unsqueeze(1) * self.current_prototypes[exist_label] + batch_prototypes[exist_label])


def compute_prototype(seg,features,classes):
    max_class = max(classes)
    out = torch.zeros((max_class+1,features.size(1))).to(seg.device)
    B,H,W = seg.shape
    features = F.interpolate(features, size=(H, W), mode='bilinear', align_corners=True)
    seg = seg.view(-1)
    features = features.permute(0,3,1,2).contiguous().view(B*H*W, -1)
    for c in classes:
        selected_features = features[seg==c]
        if len(selected_features) > 0:
            out[c] = selected_features.mean(dim=0)
    return out