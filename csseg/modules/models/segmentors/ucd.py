'''
Function:
    Implementation of UCDSegmentor
Author:
    Zhenchao Jin
'''
from .mib import MIBSegmentor


'''UCDSegmentor'''
class UCDSegmentor(MIBSegmentor):
    def __init__(self, selected_indices=(0,), num_known_classes_list=[], align_corners=False, encoder_cfg={}, decoder_cfg={}):
        super(UCDSegmentor, self).__init__(
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
        outputs = {'seg_logits': seg_logits, 'decoder_outputs': self.attention(decoder_outputs)}
        # return
        return outputs
    '''attention'''
    def attention(self, x):
        attn = torch.sum(x ** 2, dim=1)
        for i in range(attn.shape[0]):
            attn[i] = attn[i] / torch.norm(attn[i])
        attn = torch.unsqueeze(attn, 1)
        x = attn.detach() * x
        return x