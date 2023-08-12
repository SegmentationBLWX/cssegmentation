'''
Function:
    Implementation of BuildDecoder
Author:
    Zhenchao Jin
'''
import copy
from .iltaspphead import ILTASPPHead
from .mibaspphead import MIBASPPHead
from .rcilaspphead import RCILASPPHead


'''BuildDecoder'''
def BuildDecoder(decoder_cfg):
    decoder_cfg = copy.deepcopy(decoder_cfg)
    # supported decoders
    supported_decoders = {
        'ILTASPPHead': ILTASPPHead, 'MIBASPPHead': MIBASPPHead, 'RCILASPPHead': RCILASPPHead, 
    }
    # parse
    decoder_type = decoder_cfg.pop('type')
    decoder = supported_decoders[decoder_type](**decoder_cfg)
    # return
    return decoder