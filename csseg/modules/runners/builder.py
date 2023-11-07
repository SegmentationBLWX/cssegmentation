'''
Function:
    Implementation of BuildRunner and RunnerBuilder
Author:
    Zhenchao Jin
'''
from .ilt import ILTRunner
from .mib import MIBRunner
from .caf import CAFRunner
from .sdr import SDRRunner
from .plop import PLOPRunner
from .rcil import RCILRunner
from .ucd import UCDMIBRunner
from .reminder import REMINDERRunner
from ..utils import BaseModuleBuilder


'''RunnerBuilder'''
class RunnerBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'UCDMIBRunner': UCDMIBRunner, 'ILTRunner': ILTRunner, 'MIBRunner': MIBRunner, 'PLOPRunner': PLOPRunner,
        'RCILRunner': RCILRunner, 'REMINDERRunner': REMINDERRunner, 'CAFRunner': CAFRunner, 'SDRRunner': SDRRunner,
    }
    '''build'''
    def build(self, mode, cmd_args, runner_cfg):
        module_cfg = {
            'mode': mode, 'cmd_args': cmd_args, 'runner_cfg': runner_cfg, 'type': runner_cfg['type'],
        }
        return super().build(module_cfg)


'''BuildRunner'''
BuildRunner = RunnerBuilder().build