'''
Function:
    Implementation of "PLOP: Learning without Forgetting for Continual Semantic Segmentation"
Author:
    Zhenchao Jin
'''
from .base import BaseRunner


'''PLOPRunner'''
class PLOPRunner(BaseRunner):
    def __init__(self, cmd_args, runner_cfg):
        super(PLOPRunner, self).__init__(cmd_args, runner_cfg)
    '''train'''
    def train(self):
        self.segmentor.train()
    '''test'''
    def test(self):
        self.segmentor.eval()