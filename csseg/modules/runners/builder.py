'''
Function:
    Implementation of BuildRunner
Author:
    Zhenchao Jin
'''
import copy
from .ilt import ILTRunner
from .mib import MIBRunner
from .caf import CAFRunner
from .sdr import SDRRunner
from .plop import PLOPRunner
from .rcil import RCILRunner
from .ucd import UCDMIBRunner
from .reminder import REMINDERRunner


'''BuildRunner'''
def BuildRunner(mode, cmd_args, runner_cfg):
    runner_cfg = copy.deepcopy(runner_cfg)
    # supported runners
    supported_runners = {
        'UCDMIBRunner': UCDMIBRunner, 'ILTRunner': ILTRunner, 'MIBRunner': MIBRunner, 'PLOPRunner': PLOPRunner,
        'RCILRunner': RCILRunner, 'REMINDERRunner': REMINDERRunner, 'CAFRunner': CAFRunner, 'SDRRunner': SDRRunner,
    }
    # parse
    runner_type = runner_cfg.pop('type')
    runner = supported_runners[runner_type](mode=mode, cmd_args=cmd_args, runner_cfg=runner_cfg)
    # return
    return runner