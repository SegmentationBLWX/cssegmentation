'''
Function:
    Implementation of BuildRunner
Author:
    Zhenchao Jin
'''
import copy


'''BuildRunner'''
def BuildRunner(runner_cfg):
    runner_cfg = copy.deepcopy(runner_cfg)
    # supported runners
    supported_runners = {

    }
    # parse
    runner_type = runner_cfg.pop('type')
    runner = supported_runners[runner_type](**runner_cfg)
    # return
    return runner