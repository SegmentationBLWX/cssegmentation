'''
Function:
    Implementation of BuildRunner
Author:
    Zhenchao Jin
'''
import copy
from .plop import PLOPRunner
from .rcil import RCILRunner


'''BuildRunner'''
def BuildRunner(cmd_args, runner_cfg):
    runner_cfg = copy.deepcopy(runner_cfg)
    # supported runners
    supported_runners = {
        'PLOPRunner': PLOPRunner,
        'RCILRunner': RCILRunner,
    }
    # parse
    runner_type = runner_cfg.pop('type')
    runner = supported_runners[runner_type](cmd_args=cmd_args, runner_cfg=runner_cfg)
    # return
    return runner