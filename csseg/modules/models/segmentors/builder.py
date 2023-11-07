'''
Function:
    Implementation of BuildSegmentor and SegmentorBuilder
Author:
    Zhenchao Jin
'''
from .ilt import ILTSegmentor
from .ucd import UCDSegmentor
from .mib import MIBSegmentor
from .base import BaseSegmentor
from .plop import PLOPSegmentor
from ...utils import BaseModuleBuilder


'''SegmentorBuilder'''
class SegmentorBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'MIBSegmentor': MIBSegmentor, 'ILTSegmentor': ILTSegmentor, 'BaseSegmentor': BaseSegmentor, 'PLOPSegmentor': PLOPSegmentor, 'UCDSegmentor': UCDSegmentor,
    }
    '''build'''
    def build(self, segmentor_cfg):
        return super().build(segmentor_cfg)


'''BuildSegmentor'''
BuildSegmentor = SegmentorBuilder().build