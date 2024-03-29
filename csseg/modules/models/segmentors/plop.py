'''
Function:
    Implementation of PLOPSegmentor
Author:
    Zhenchao Jin
'''
import torch
from .mib import MIBSegmentor


'''PLOPSegmentor'''
class PLOPSegmentor(MIBSegmentor):
    def __init__(self, selected_indices=(0, 1, 2, 3), num_known_classes_list=[], align_corners=False, encoder_cfg={}, decoder_cfg={}):
        super(PLOPSegmentor, self).__init__(
            selected_indices=selected_indices, num_known_classes_list=num_known_classes_list, 
            align_corners=align_corners, encoder_cfg=encoder_cfg, decoder_cfg=decoder_cfg,
        )
    '''forward'''
    def forward(self, x):
        # feed to encoder
        encoder_outputs, distillation_feats = self.encoder(x)
        # select encoder outputs
        selected_feats = self.transforminputs(encoder_outputs, self.selected_indices)
        # feed to decoder
        decoder_outputs = self.decoder(selected_feats)
        # feed to classifier
        seg_logits = [conv(decoder_outputs) for conv in self.convs_cls]
        seg_logits = torch.cat(seg_logits, dim=1)
        # construct outputs
        outputs = {'seg_logits': seg_logits, 'distillation_feats': list(distillation_feats) + [decoder_outputs]}
        # return
        return outputs