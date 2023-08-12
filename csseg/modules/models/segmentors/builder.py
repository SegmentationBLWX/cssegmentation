'''
Function:
    Implementation of BuildSegmentor
Author:
    Zhenchao Jin
'''
import copy
from .ilt import ILTSegmentor
from .ucd import UCDSegmentor
from .mib import MIBSegmentor
from .base import BaseSegmentor
from .plop import PLOPSegmentor


'''BuildSegmentor'''
def BuildSegmentor(segmentor_cfg):
    segmentor_cfg = copy.deepcopy(segmentor_cfg)
    # supported segmentors
    supported_segmentors = {
        'MIBSegmentor': MIBSegmentor, 'ILTSegmentor': ILTSegmentor, 'BaseSegmentor': BaseSegmentor, 'PLOPSegmentor': PLOPSegmentor, 'UCDSegmentor': UCDSegmentor,
    }
    # parse
    segmentor_type = segmentor_cfg.pop('type')
    segmentor = supported_segmentors[segmentor_type](**segmentor_cfg)
    # return
    return segmentor