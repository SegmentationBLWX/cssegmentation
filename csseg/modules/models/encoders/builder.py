'''
Function:
    Implementation of EncoderBuilder and BuildEncoder
Author:
    Zhenchao Jin
'''
from .resnet import ResNet
from .resnetilt import ResNetILT
from .resnetplop import ResNetPLOP
from .resnetrcil import ResNetRCIL
from ...utils import BaseModuleBuilder


'''EncoderBuilder'''
class EncoderBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'ResNet': ResNet, 'ResNetILT': ResNetILT, 'ResNetPLOP': ResNetPLOP, 'ResNetRCIL': ResNetRCIL,
    }
    '''build'''
    def build(self, encoder_cfg):
        return super().build(encoder_cfg)


'''BuildEncoder'''
BuildEncoder = EncoderBuilder().build