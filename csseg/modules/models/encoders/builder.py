'''
Function:
    Implementation of BuildEncoder
Author:
    Zhenchao Jin
'''
import copy
from .resnet import ResNet
from .resnetmib import ResNetMIB
from .resnetplop import ResNetPLOP


'''BuildEncoder'''
def BuildEncoder(encoder_cfg):
    encoder_cfg = copy.deepcopy(encoder_cfg)
    # supported encoders
    supported_encoders = {
        'ResNet': ResNet,
        'ResNetMIB': ResNetMIB,
        'ResNetPLOP': ResNetPLOP,
    }
    # parse
    encoder_type = encoder_cfg.pop('type')
    encoder = supported_encoders[encoder_type](**encoder_cfg)
    # return
    return encoder