'''
Function:
    Implementation of PLOPSegmentor
Author:
    Zhenchao Jin
'''
from .base import BaseSegmentor


'''PLOPSegmentor'''
class PLOPSegmentor(BaseSegmentor):
    def __init__(self, mode, seleced_indices=(0, 1, 2, 3), num_classes_list=[], encoder_cfg={}, decoder_cfg={}):
        super(BaseSegmentor, self).__init__(mode)