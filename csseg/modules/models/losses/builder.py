'''
Function:
    Implementation of BuildLoss and LossBuilder
Author:
    Zhenchao Jin
'''
from .mseloss import MSELoss
from .klloss import KLDivLoss
from ...utils import BaseModuleBuilder
from .csloss import CosineSimilarityLoss
from .celoss import CrossEntropyLoss, MIBUnbiasedCrossEntropyLoss


'''LossBuilder'''
class LossBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'MSELoss': MSELoss, 'KLDivLoss': KLDivLoss, 'CrossEntropyLoss': CrossEntropyLoss, 'CosineSimilarityLoss': CosineSimilarityLoss,
        'MIBUnbiasedCrossEntropyLoss': MIBUnbiasedCrossEntropyLoss,
    }
    '''build'''
    def build(self, loss_cfg):
        return super().build(loss_cfg)


'''BuildLoss'''
BuildLoss = LossBuilder().build