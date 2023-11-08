'''
Function:
    Implementation of Tester
Author:
    Zhenchao Jin
'''
import torch
import warnings
import argparse
import torch.distributed as dist
from configs import BuildConfig
from modules import BuildRunner, loadckpts
warnings.filterwarnings('ignore')


'''parsecmdargs'''
def parsecmdargs():
    parser = argparse.ArgumentParser(description='CSSegmentation: An Open Source Continual Semantic Segmentation Toolbox Based on PyTorch.')
    parser.add_argument('--local_rank', '--local-rank', dest='local_rank', help='node rank for distributed training.', default=0, type=int)
    parser.add_argument('--nproc_per_node', dest='nproc_per_node', help='number of processes per node.', default=4, type=int)
    parser.add_argument('--cfgfilepath', dest='cfgfilepath', help='config file path you want to load.', type=str, required=True)
    parser.add_argument('--ckptspath', dest='ckptspath', help='checkpoints path you want to load.', type=str, required=True)
    cmd_args = parser.parse_args()
    return cmd_args


'''Tester'''
class Tester():
    def __init__(self, cmd_args):
        self.cmd_args = cmd_args
        self.cfg = BuildConfig(cmd_args.cfgfilepath)[0]
    '''start'''
    def start(self):
        # initialize
        assert torch.cuda.is_available(), 'cuda is not available'
        cmd_args, runner_cfg = self.cmd_args, self.cfg.RUNNER_CFG
        dist.init_process_group(backend=runner_cfg['parallel_cfg']['backend'], init_method=runner_cfg['parallel_cfg']['init_method'])
        torch.cuda.set_device(cmd_args.local_rank)
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = runner_cfg['benchmark']
        # load ckpts
        ckpts = loadckpts(cmd_args.ckptspath)
        runner_cfg['task_id'] = ckpts['task_id']
        runner_client = BuildRunner(mode='TEST', cmd_args=cmd_args, runner_cfg=runner_cfg)
        runner_client.segmentor.load_state_dict(ckpts['segmentor'], strict=True)
        # start to test and print results
        results = runner_client.test(cur_epoch=ckpts['cur_epoch'])
        if cmd_args.local_rank == 0:
            runner_client.logger_handle.info(results)


'''main'''
if __name__ == '__main__':
    cmd_args = parsecmdargs()
    tester_client = Tester(cmd_args=cmd_args)
    tester_client.start()