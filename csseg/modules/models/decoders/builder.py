'''
Function:
    Implementation of BuildDecoder
Author:
    Zhenchao Jin
'''
from .aspphead import ASPPHead
from .rcilaspphead import RCILASPPHead
from ...utils import BaseModuleBuilder


'''DecoderBuilder'''
class DecoderBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'ASPPHead': ASPPHead, 'RCILASPPHead': RCILASPPHead, 
    }
    '''build'''
    def build(self, decoder_cfg):
        return super().build(decoder_cfg)


'''BuildDecoder'''
BuildDecoder = DecoderBuilder().build