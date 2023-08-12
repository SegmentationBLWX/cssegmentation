'''
Function:
    Implementation of ResNetILT
Author:
    Zhenchao Jin
'''
from .resnet import ResNet, BasicBlock, Bottleneck


'''BasicBlockILT'''
class BasicBlockILT(BasicBlock):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, norm_cfg=None, act_cfg=None, shortcut_norm_cfg=None, shortcut_act_cfg=None):
        super(BasicBlockILT, self).__init__(
            inplanes=inplanes, planes=planes, stride=stride, dilation=dilation, downsample=downsample, 
            norm_cfg=norm_cfg, act_cfg=act_cfg, shortcut_norm_cfg=shortcut_norm_cfg, shortcut_act_cfg=shortcut_act_cfg,
        )


'''BottleneckILT'''
class BottleneckILT(Bottleneck):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, norm_cfg=None, act_cfg=None, shortcut_norm_cfg=None, shortcut_act_cfg=None):
        super(BottleneckILT, self).__init__(
            inplanes=inplanes, planes=planes, stride=stride, dilation=dilation, downsample=downsample, 
            norm_cfg=norm_cfg, act_cfg=act_cfg, shortcut_norm_cfg=shortcut_norm_cfg, shortcut_act_cfg=shortcut_act_cfg,
        )


'''ResNetILT'''
class ResNetILT(ResNet):
    def __init__(self, in_channels=3, base_channels=64, stem_channels=64, depth=101, outstride=16, contract_dilation=False, deep_stem=False, 
                 out_indices=(3,), use_avg_for_downsample=False, norm_cfg={'type': 'InPlaceABNSync', 'activation': 'leaky_relu', 'activation_param': 0.01}, 
                 act_cfg=None,  pretrained=True, pretrained_model_path=None, user_defined_block=None, use_inplaceabn_style=True):
        if user_defined_block is None:
            user_defined_block = BasicBlockILT if depth in [18, 34] else BottleneckILT
        super(ResNetILT, self).__init__(
            in_channels=in_channels, base_channels=base_channels, stem_channels=stem_channels, depth=depth, outstride=outstride, 
            contract_dilation=contract_dilation, deep_stem=deep_stem, out_indices=out_indices, use_avg_for_downsample=use_avg_for_downsample, 
            norm_cfg=norm_cfg, act_cfg=act_cfg, pretrained=pretrained, pretrained_model_path=pretrained_model_path, user_defined_block=user_defined_block,
            use_inplaceabn_style=use_inplaceabn_style,
        )