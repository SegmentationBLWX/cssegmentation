'''
Function:
    Implementation of ILTASPPHead
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..encoders import BuildNormalization


'''ILTASPPHead'''
class ILTASPPHead(nn.Module):
    def __init__(self, in_channels, feats_channels, out_channels, dilations, pooling_size=32, norm_cfg=None):
        super(ILTASPPHead, self).__init__()
        # assert
        assert norm_cfg['type'] in ['ABN', 'InPlaceABN', 'InPlaceABNSync']
        # set attributes
        self.in_channels = in_channels
        self.feats_channels = feats_channels
        self.out_channels = out_channels
        self.pooling_size = (pooling_size, pooling_size) if isinstance(pooling_size, int) else pooling_size
        # parallel convolutions
        self.map_convs = nn.ModuleList()
        for idx, dilation in enumerate(dilations):
            if dilation == 1:
                conv = nn.Conv2d(in_channels, feats_channels, kernel_size=1, stride=1, padding=0, dilation=dilation, bias=False)
            else:
                conv = nn.Conv2d(in_channels, feats_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
            self.map_convs.append(conv)
        self.map_bn = BuildNormalization(placeholder=feats_channels * len(dilations), norm_cfg=norm_cfg)
        # global branch
        self.global_pooling_conv = nn.Conv2d(in_channels, feats_channels, 1, bias=False)
        self.global_pooling_bn = BuildNormalization(placeholder=feats_channels, norm_cfg=norm_cfg)
        self.pool_red_conv = nn.Conv2d(feats_channels, out_channels, 1, bias=False)
        # output project
        self.red_conv = nn.Conv2d(feats_channels * 4, out_channels, 1, bias=False)
        self.red_bn = BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg)
        # initialize parameters
        self.initparams(self.red_bn.activation, self.red_bn.activation_param)
    '''initparams'''
    def initparams(self, nonlinearity, param=None):
        gain = nn.init.calculate_gain(nonlinearity, param)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_normal_(module.weight.data, gain)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    '''forward'''
    def forward(self, x):
        # feed to parallel convolutions
        out = torch.cat([m(x) for m in self.map_convs], dim=1)
        out = self.map_bn(out)
        out = self.red_conv(out)
        # feed to global branch
        pool = self.globalpooling(x)
        pool = self.global_pooling_conv(pool)
        pool = self.global_pooling_bn(pool)
        pool = self.pool_red_conv(pool)
        if self.training or self.pooling_size is None:
            pool = pool.repeat(1, 1, x.size(2), x.size(3))
        # shortcut
        out += pool
        out = self.red_bn(out)
        # return
        return out
    '''globalpooling'''
    def globalpooling(self, x):
        if self.training or self.pooling_size is None:
            global_feats = x.view(x.size(0), x.size(1), -1).mean(dim=-1)
            global_feats = global_feats.view(x.size(0), x.size(1), 1, 1)
        else:
            pooling_size = (min(self.pooling_size[0], x.shape[2]), min(self.pooling_size[1], x.shape[3]))
            padding = (
                (pooling_size[1] - 1) // 2, (pooling_size[1] - 1) // 2 if pooling_size[1] % 2 == 1 else (pooling_size[1] - 1) // 2 + 1,
                (pooling_size[0] - 1) // 2, (pooling_size[0] - 1) // 2 if pooling_size[0] % 2 == 1 else (pooling_size[0] - 1) // 2 + 1,
            )
            global_feats = F.avg_pool2d(x, pooling_size, stride=1)
            global_feats = F.pad(global_feats, pad=padding, mode='replicate')
        return global_feats