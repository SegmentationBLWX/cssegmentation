'''
Function:
    Implementation of MIBSegmentor
Author:
    Zhenchao Jin
'''
import torch
from .base import BaseSegmentor


'''MIBSegmentor'''
class MIBSegmentor(BaseSegmentor):
    def __init__(self, selected_indices=(0,), num_known_classes_list=[], align_corners=False, encoder_cfg={}, decoder_cfg={}):
        super(MIBSegmentor, self).__init__(
            selected_indices=selected_indices, num_known_classes_list=num_known_classes_list, 
            align_corners=align_corners, encoder_cfg=encoder_cfg, decoder_cfg=decoder_cfg,
        )
    '''initaddedclassifier'''
    def initaddedclassifier(self, device=None):
        conv_cls = self.convs_cls[-1]
        imprinting_w = self.convs_cls[0].weight[0]
        bkg_bias = self.convs_cls[0].bias[0]
        bias_diff = torch.log(torch.FloatTensor([self.num_known_classes_list[-1] + 1])).to(device)
        new_bias = (bkg_bias - bias_diff)
        conv_cls.weight.data.copy_(imprinting_w)
        conv_cls.bias.data.copy_(new_bias)
        self.convs_cls[0].bias[0].data.copy_(new_bias.squeeze(0))