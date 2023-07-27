'''
Function:
    Implementation of BuildDecoder
Author:
    Zhenchao Jin
'''
import copy
from .rcilaspphead import RCILASPPHead
from .baseclaspphead import BASECLASPPHead


'''BuildDecoder'''
def BuildDecoder(decoder_cfg):
    decoder_cfg = copy.deepcopy(decoder_cfg)
    # supported decoders
    supported_decoders = {
        'BASECLASPPHead': BASECLASPPHead, 'RCILASPPHead': RCILASPPHead,
    }
    # parse
    decoder_type = decoder_cfg.pop('type')
    decoder = supported_decoders[decoder_type](**decoder_cfg)
    # return
    return decoder