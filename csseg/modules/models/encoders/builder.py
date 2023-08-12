'''
Function:
    Implementation of BuildEncoder
Author:
    Zhenchao Jin
'''
import copy
from .resnet import ResNet
from .resnetilt import ResNetILT
from .resnetplop import ResNetPLOP
from .resnetrcil import ResNetRCIL


'''BuildEncoder'''
def BuildEncoder(encoder_cfg):
    encoder_cfg = copy.deepcopy(encoder_cfg)
    # supported encoders
    supported_encoders = {
        'ResNet': ResNet, 'ResNetILT': ResNetILT, 'ResNetPLOP': ResNetPLOP, 'ResNetRCIL': ResNetRCIL,
    }
    # parse
    encoder_type = encoder_cfg.pop('type')
    encoder = supported_encoders[encoder_type](**encoder_cfg)
    # return
    return encoder