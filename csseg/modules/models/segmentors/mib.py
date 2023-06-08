'''
Function:
    Implementation of MIBSegmentor
Author:
    Zhenchao Jin
'''
import torch
import torch.nn.functional as F
from .base import BaseSegmentor


'''MIBSegmentor'''
class MIBSegmentor(BaseSegmentor):
    def __init__(self, selected_indices=(0,), num_known_classes_list=[], align_corners=False, encoder_cfg={}, decoder_cfg={}):
        super(MIBSegmentor, self).__init__(
            selected_indices=selected_indices, num_known_classes_list=num_known_classes_list, 
            align_corners=align_corners, encoder_cfg=encoder_cfg, decoder_cfg=decoder_cfg,
        )
    '''forward'''
    def forward(self, x):
        img_size = x.shape[2:]
        # feed to encoder
        encoder_outputs = self.encoder(x)
        # select encoder outputs
        selected_feats = self.transforminputs(encoder_outputs, self.selected_indices)
        # feed to decoder
        decoder_outputs = self.decoder(selected_feats)
        # feed to classifier
        seg_logits = [conv(decoder_outputs) for conv in self.convs_cls]
        seg_logits = torch.cat(seg_logits, dim=1)
        # construct outputs
        outputs = {'seg_logits': seg_logits}
        # return
        return outputs
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