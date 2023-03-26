'''
Function:
    Implementation of BuildScheduler
Author:
    Zhenchao Jin
'''
import copy
from .poly import PolyScheduler


'''BuildScheduler'''
def BuildScheduler(scheduler_cfg):
    scheduler_cfg = copy.deepcopy(scheduler_cfg)
    # supported schedulers
    supported_schedulers = {
        'PolyScheduler': PolyScheduler
    }
    # parse
    scheduler_type = scheduler_cfg.pop('type')
    scheduler = supported_schedulers[scheduler_type](**scheduler_cfg)
    # return
    return scheduler