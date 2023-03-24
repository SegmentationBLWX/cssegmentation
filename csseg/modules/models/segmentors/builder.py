'''
Function:
    Implementation of BuildSegmentor
Author:
    Zhenchao Jin
'''
import copy
from .base import BaseSegmentor


'''BuildSegmentor'''
def BuildSegmentor(segmentor_cfg):
    segmentor_cfg = copy.deepcopy(segmentor_cfg)
    # supported segmentors
    supported_segmentors = {
        'BaseSegmentor': BaseSegmentor,
    }
    # parse
    segmentor_type = segmentor_cfg.pop('type')
    segmentor = supported_segmentors[segmentor_type](**segmentor_cfg)
    # return
    return segmentor