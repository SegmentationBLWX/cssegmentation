'''
Function:
    Implementation of BaseRunner
Author:
    Zhenchao Jin
'''
import os
import copy
import pickle
from apex import amp
from ..datasets import BuildDataset
from ..models import BuildSegmentor, BuildOptimizer, BuildScheduler
from ..utils import Logger, touchdir, loadckpts, saveckpts, saveaspickle
from ..parallel import BuildDistributedDataloader, BuildDistributedModel


'''BaseRunner'''
class BaseRunner():
    def __init__(self, mode, cmd_args, runner_cfg):
        # set attributes
        self.best_score = 0
        self.cmd_args = cmd_args
        self.runner_cfg = runner_cfg
        self.device = torch.device(cmd_args.local_rank)
        self.root_work_dir = runner_cfg['work_dir']
        self.task_work_dir = os.path.join(runner_cfg['work_dir'], f'task_{runner_cfg["task_id"]}')
        self.save_interval_epochs = runner_cfg['save_interval_epochs']
        self.eval_interval_epochs = runner_cfg['eval_interval_epochs']
        self.log_interval_iterations = runner_cfg['log_interval_iterations']
        self.choose_best_segmentor_by_metric = runner_cfg['choose_best_segmentor_by_metric']
        # build logger handle
        self.logger_handle = Logger(logfilepath=runner_cfg['logfilepath'])
        # build workdir
        touchdir(dirname=self.root_work_dir)
        touchdir(dirname=self.task_work_dir)
        # build datasets
        dataset_cfg = runner_cfg['DATASET_CFG']
        train_set = BuildDataset(mode='TRAIN', dataset_cfg=dataset_cfg)
        test_set = BuildDataset(mode='TEST', dataset_cfg=dataset_cfg)
        # build dataloaders
        dataloader_cfg = runner_cfg['DATALOADER_CFG']
        self.train_loader = BuildDistributedDataloader(dataset=train_set, dataloader_cfg=dataloader_cfg)
        self.test_loader = BuildDistributedDataloader(dataset=test_set, dataloader_cfg=dataloader_cfg)
        # build segmentor
        segmentor_cfg = runner_cfg['SEGMENTOR_CFG']
        segmentor_cfg['mode'] = mode
        self.segmentor = BuildSegmentor(segmentor_cfg=segmentor_cfg)
        if runner_cfg['task_id'] > 1:
            history_segmentor_cfg = copy.deepcopy(segmentor_cfg)
            history_segmentor_cfg['mode'] = 'TEST'
            history_segmentor_cfg['num_classes_list'] = segmentor_cfg['num_classes_list'][:-1]
            self.history_segmentor = BuildSegmentor(segmentor_cfg=history_segmentor_cfg)
        else:
            self.history_segmentor = None
        # build optimizer
        optimizer_cfg = runner_cfg['OPTIMIZER_CFG']
        self.optimizer = BuildOptimizer(model=self.segmentor, optimizer_cfg=optimizer_cfg)
        # build scheduler
        scheduler_cfg = runner_cfg['SCHEDULER_CFG']
        self.scheduler = BuildScheduler(scheduler_cfg=scheduler_cfg)
        # parallel segmentor
        parallel_cfg = runner_cfg['PARALLEL_CFG']
        if self.history_segmentor is None:
            self.segmentor, self.optimizer = amp.initialize(
                self.segmentor.to(self.device), self.optimizer, opt_level=parallel_cfg['opt_level']
            )
        else:
            [self.segmentor, self.history_segmentor], self.optimizer = amp.initialize(
                [self.segmentor.to(device), self.history_segmentor.to(device)], self.optimizer, opt_level=parallel_cfg['opt_level']
            )
            self.history_segmentor = BuildDistributedModel(model=self.history_segmentor, model_cfg={})
        self.segmentor = BuildDistributedModel(model=self.segmentor, model_cfg={'delay_allreduce': True})
        # load history checkpoints
        if self.history_segmentor is not None:
            history_task_work_dir = os.path.join(runner_cfg['work_dir'], f'task_{runner_cfg["task_id"] - 1}')
            ckpts = loadckpts(os.path.join(history_task_work_dir, 'best.pth'))
            self.segmentor.load_state_dict(ckpts['segmentor'], strict=False)
            if hasattr(self.segmentor.module, 'initaddedclassifier'):
                self.segmentor.module.initaddedclassifier(device=self.device)
            self.history_segmentor.load_state_dict(ckpts['segmentor'], strict=True)
            for param in self.history_segmentor.parameters():
                param.requires_grad = False
            self.history_segmentor.eval()
        # load current checkpoints
        if os.path.exists(os.path.join(self.task_work_dir, 'latest.pth')):
            ckpts = loadckpts(os.path.join(self.task_work_dir, 'latest.pth'))
            self.segmentor.load_state_dict(ckpts['segmentor'], strict=True)
            self.optimizer.load_state_dict(ckpts['optimizer'])
            self.scheduler.load(state_dict=ckpts)
            self.best_score = ckpts['best_score']
    '''start'''
    def start(self):
        if self.cmd_args.local_rank == 0:
            self.logger_handle.info(f'Load Config From: {self.cmd_args.cfgfilepath}')
            self.logger_handle.info(f'Config Details: \n{self.runner_cfg}')
        for cur_epoch in range(self.scheduler.cur_epoch, self.scheduler.max_epochs+1):
            self.train()
            if ((cur_epoch % self.save_interval_epochs == 0) or (cur_epoch == self.scheduler.max_epochs)) and (self.cmd_args.local_rank == 0):
                ckpt_path = os.path.join(self.task_work_dir, f'epoch_{cur_epoch}.pth')
                saveckpts(ckpts=self.state(), savepath=ckpt_path)
                os.symlink(ckpt_path, os.path.join(self.task_work_dir, 'latest.pth'))
            if (cur_epoch % self.eval_interval_epochs == 0) or (cur_epoch == self.scheduler.max_epochs):
                results = self.test()
                if self.cmd_args.local_rank == 0:
                    ckpt_path = os.path.join(self.task_work_dir, f'epoch_{cur_epoch}.pth')
                    if self.best_score <= results[self.choose_best_segmentor_by_metric]:
                        self.best_score = results[self.choose_best_segmentor_by_metric]
                        os.symlink(ckpt_path, os.path.join(self.task_work_dir, 'best.pth'))
                        saveaspickle(results, os.path.join(self.task_work_dir, 'best.pkl'))
    '''train'''
    def train(self):
        raise NotImplementedError('not to be implemented')
    '''test'''
    def test(self):
        raise NotImplementedError('not to be implemented')
    '''state'''
    def state(self):
        state_dict = self.scheduler.state()
        state_dict.update({
            'best_score': self.best_score,
            'segmentor': self.segmentor.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iters_per_epoch': len(self.train_loader), 
        })
        return state_dict