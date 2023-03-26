'''
Function:
    Implementation of BaseSegmentor
Author:
    Zhenchao Jin
'''
import torch
import numbers
import collections
import torch.nn as nn
import torch.nn.functional as F
from ..losses import BuildLoss
from ..encoders import BuildEncoder
from ..decoders import BuildDecoder


'''BaseSegmentor'''
class BaseSegmentor(nn.Module):
    def __init__(self, mode, seleced_indices=(0, 1, 2, 3), num_classes_list=[], align_corners=False, encoder_cfg={}, decoder_cfg={}):
        # assert
        assert mode in ['TRAIN', 'TEST']
        assert isinstance(seleced_indices, (numbers.Number, collections.Sequence))
        # set attributes
        self.mode = mode
        self.align_corners = align_corners
        self.seleced_indices = seleced_indices
        # build encoder and decoder
        self.encoder = BuildEncoder(encoder_cfg)
        self.decoder = BuildDecoder(decoder_cfg)
        # build classifier
        self.convs_cls = nn.ModuleList([
            nn.Conv2d(self.decoder.out_channels, num_classes, kernel_size=1, stride=1, padding=0) for num_classes in num_classes_list
        ])
    '''forward'''
    def forward(self, x, seg_targets=None, losses_cfgs=None):
        img_size = x.shape[2:]
        # feed to encoder
        encoder_outputs = self.encoder(x)
        # select encoder outputs
        selected_feats = self.transforminputs(encoder_outputs, self.seleced_indices)
        # feed to decoder
        decoder_outputs = self.decoder(selected_feats)
        # feed to classifier
        seg_logits = [conv(decoder_outputs) for conv in self.convs_cls]
        seg_logits = torch.cat(seg_logits, dim=1)
        # construct outputs
        outputs = {'seg_logits': seg_logits}
        # calculate segmentation losses if `mode` is 'TRAIN'
        if self.mode == 'TRAIN':
            loss_total, losses_log_dict = self.calculatelosses(seg_logits, seg_targets, losses_cfgs)
            outputs.update({
                'loss_total': loss_total, 'losses_log_dict': losses_log_dict
            })
        # return
        return outputs
    '''calculateloss'''
    def calculateloss(self, seg_logits, seg_targets, losses_cfg):
        loss = 0
        for loss_type, loss_cfg in losses_cfg.items():
            loss_cfg['type'] = loss_type
            loss += BuildLoss(loss_cfg)(prediction=seg_logits, target=seg_targets)
        return loss
    '''calculatelosses'''
    def calculatelosses(self, seg_logits, seg_targets, losses_cfgs):
        # iter to calculate losses
        losses_log_dict, loss_total = {}, 0
        for losses_name, losses_cfg in losses_cfgs.items():
            losses_log_dict[losses_name] = self.calculateloss(
                seg_logits=seg_logits, seg_targets=seg_targets, losses_cfg=losses_cfg
            )
            loss_total += losses_log_dict[losses_name]
        losses_log_dict.update({'loss_total': loss_total})
        # syn losses_log_dict
        for key, value in losses_log_dict.items():
            if dist.is_available() and dist.is_initialized():
                value = value.data.clone()
                dist.all_reduce(value.div_(dist.get_world_size()))
                losses_log_dict[key] = value.item()
        # return
        return loss_total, losses_log_dict
    '''transforminputs'''
    def transforminputs(self, inputs, seleced_indices):
        if isinstance(seleced_indices, numbers.Number):
            seleced_indices = [seleced_indices]
        outputs = [inputs[idx] for idx in seleced_indices]
        return outputs if len(seleced_indices) > 1 else outputs[0]