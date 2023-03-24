'''
Function:
    Implementation of ResNetPLOP
Author:
    Zhenchao Jin
'''
from .resnet import ResNet, BasicBlock, Bottleneck


'''BasicBlockPLOP'''
class BasicBlockPLOP(BasicBlock):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, norm_cfg=None, act_cfg=None):
        super(BasicBlockPLOP, self).__init__(
            inplanes=inplanes, planes=planes, stride=stride, dilation=dilation, downsample=downsample, 
            norm_cfg=norm_cfg, act_cfg=act_cfg,
        )
    '''forward'''
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None: identity = self.downsample(x)
        out += identity
        attention = out
        out = self.relu(out)
        return out, attention


'''BottleneckPLOP'''
class BottleneckPLOP(Bottleneck):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, norm_cfg=None, act_cfg=None):
        super(BottleneckPLOP, self).__init__(
            inplanes=inplanes, planes=planes, stride=stride, dilation=dilation, downsample=downsample, 
            norm_cfg=norm_cfg, act_cfg=act_cfg,
        )
    '''forward'''
    def forward(self, x):
        if isinstance(x, tuple): x = x[0]
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None: identity = self.downsample(x)
        out += identity
        attention = out
        out = self.relu(out)
        return out, attention


'''ResNetPLOP'''
class ResNetPLOP(ResNet):
    def __init__(self, in_channels=3, base_channels=64, stem_channels=64, depth=101, outstride=16, contract_dilation=False, deep_stem=False, 
                 out_indices=(0, 1, 2, 3), use_avg_for_downsample=False, norm_cfg={'type': 'InPlaceABNSync', 'activation': 'identity'}, 
                 act_cfg={'type': 'LeakyReLU', 'inplace': True, 'negative_slope': 0.01},  pretrained=True, pretrained_model_path=None, 
                 user_defined_block=None, use_inplaceabn_style=True):
        if user_defined_block is None:
            user_defined_block = BasicBlockPLOP if depth in [18, 34] else BottleneckPLOP
        super(ResNetPLOP, self).__init__(
            in_channels=in_channels, base_channels=base_channels, stem_channels=stem_channels, depth=depth, outstride=outstride, 
            contract_dilation=contract_dilation, deep_stem=deep_stem, out_indices=out_indices, use_avg_for_downsample=use_avg_for_downsample, 
            norm_cfg=norm_cfg, act_cfg=act_cfg, pretrained=pretrained, pretrained_model_path=pretrained_model_path, user_defined_block=user_defined_block,
            use_inplaceabn_style=use_inplaceabn_style,
        )
    '''forward'''
    def forward(self, x):
        outs, attentions = [], []
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        x1, attention1 = self.layer1(x)
        x2, attention2 = self.layer2(x1)
        x3, attention3 = self.layer3(x2)
        x4, attention4 = self.layer4(x3)
        outs = []
        for i, feats in enumerate([(x1, attention1), (x2, attention2), (x3, attention3), (x4, attention4)]):
            if i in self.out_indices: 
                outs.append(feats[0])
                attentions.append(feats[1])
        return tuple(outs), tuple(attentions)