'''
Function:
    Implementation of BaseRunner
Author:
    Zhenchao Jin
'''
import os
import copy
import torch
import random
import torch.nn.functional as F
from apex import amp
from tqdm import tqdm
from ..datasets import BuildDataset, SegmentationEvaluator
from ..models import BuildSegmentor, BuildOptimizer, BuildScheduler
from ..parallel import BuildDistributedDataloader, BuildDistributedModel
from ..utils import Logger, touchdir, loadckpts, saveckpts, saveaspickle, symlink, loadpicklefile, setrandomseed


'''BaseRunner'''
class BaseRunner():
    def __init__(self, mode, cmd_args, runner_cfg):
        # assert
        assert mode in ['TRAIN', 'TEST']
        # set attributes
        self.mode = mode
        self.best_score = 0
        self.cmd_args = cmd_args
        self.runner_cfg = runner_cfg
        self.losses_cfgs = runner_cfg['segmentor_cfg']['losses_cfgs']
        self.device = torch.device(cmd_args.local_rank)
        self.root_work_dir = runner_cfg['work_dir']
        self.task_work_dir = os.path.join(runner_cfg['work_dir'], f'task_{runner_cfg["task_id"]}')
        self.save_interval_epochs = runner_cfg['save_interval_epochs']
        self.eval_interval_epochs = runner_cfg['eval_interval_epochs']
        self.log_interval_iterations = runner_cfg['log_interval_iterations']
        self.choose_best_segmentor_by_metric = runner_cfg['choose_best_segmentor_by_metric']
        self.eps = runner_cfg.get('eps', 1e-6)
        # build workdir
        touchdir(dirname=self.root_work_dir)
        touchdir(dirname=self.task_work_dir)
        # build logger handle
        self.logger_handle = Logger(logfilepath=runner_cfg['logfilepath'])
        # build datasets
        setrandomseed(runner_cfg['random_seed'])
        dataset_cfg = runner_cfg['dataset_cfg']
        train_set = BuildDataset(mode='TRAIN', task_name=runner_cfg['task_name'], task_id=runner_cfg['task_id'], dataset_cfg=dataset_cfg) if mode == 'TRAIN' else None
        test_set = BuildDataset(mode='TEST', task_name=runner_cfg['task_name'], task_id=runner_cfg['task_id'], dataset_cfg=dataset_cfg)
        assert (runner_cfg['num_total_classes'] == train_set.num_classes if mode == 'TRAIN' else True)
        assert runner_cfg['num_total_classes'] == test_set.num_classes
        random.seed(runner_cfg['random_seed'])
        # build dataloaders
        dataloader_cfg = copy.deepcopy(runner_cfg['dataloader_cfg'])
        total_train_bs_for_auto_check = dataloader_cfg.pop('total_train_bs_for_auto_check')
        auto_align_train_bs = dataloader_cfg.pop('auto_align_train_bs')
        if auto_align_train_bs:
            dataloader_cfg['train']['batch_size_per_gpu'] = total_train_bs_for_auto_check // self.cmd_args.nproc_per_node
        assert dataloader_cfg['train']['batch_size_per_gpu'] * self.cmd_args.nproc_per_node == total_train_bs_for_auto_check
        self.train_loader = BuildDistributedDataloader(dataset=train_set, dataloader_cfg=dataloader_cfg) if mode == 'TRAIN' else None
        self.test_loader = BuildDistributedDataloader(dataset=test_set, dataloader_cfg=dataloader_cfg)
        # build segmentor
        if train_set is None:
            runner_cfg['segmentor_cfg']['num_known_classes_list'] = test_set.getnumclassespertask(runner_cfg['task_name'], test_set.tasks, runner_cfg['task_id'])
        else:
            runner_cfg['segmentor_cfg']['num_known_classes_list'] = train_set.getnumclassespertask(runner_cfg['task_name'], train_set.tasks, runner_cfg['task_id'])
        segmentor_cfg = copy.deepcopy(runner_cfg['segmentor_cfg'])
        segmentor_cfg.pop('losses_cfgs')
        self.segmentor = BuildSegmentor(segmentor_cfg=segmentor_cfg)
        if runner_cfg['task_id'] > 0 and mode == 'TRAIN':
            history_segmentor_cfg = copy.deepcopy(segmentor_cfg)
            history_segmentor_cfg['num_known_classes_list'] = segmentor_cfg['num_known_classes_list'][:-1]
            self.history_segmentor = BuildSegmentor(segmentor_cfg=history_segmentor_cfg)
        else:
            self.history_segmentor = None
        # build optimizer
        if mode == 'TRAIN':
            scheduler_cfg = copy.deepcopy(runner_cfg['scheduler_cfg'])
            optimizer_cfg = scheduler_cfg['optimizer_cfg']
            optimizer_cfg['lr'] = scheduler_cfg['lr']
            self.optimizer = BuildOptimizer(model=self.segmentor, optimizer_cfg=optimizer_cfg)
            scheduler_cfg.update({
                'iters_per_epoch': len(self.train_loader), 'paramwise_cfg': optimizer_cfg['paramwise_cfg']
            })
        else:
            self.optimizer = None
        # build scheduler
        self.scheduler = BuildScheduler(optimizer=self.optimizer, scheduler_cfg=scheduler_cfg) if mode == 'TRAIN' else None
        # parallel segmentor
        parallel_cfg = runner_cfg['parallel_cfg']
        if self.history_segmentor is None and mode == 'TRAIN':
            self.segmentor, self.optimizer = amp.initialize(
                self.segmentor.to(self.device), self.optimizer, opt_level=parallel_cfg['opt_level']
            )
        elif mode == 'TRAIN':
            [self.segmentor, self.history_segmentor], self.optimizer = amp.initialize(
                [self.segmentor.to(self.device), self.history_segmentor.to(self.device)], self.optimizer, opt_level=parallel_cfg['opt_level']
            )
            self.history_segmentor = BuildDistributedModel(model=self.history_segmentor, model_cfg={})
        else:
            self.segmentor = self.segmentor.to(self.device)
        self.segmentor = BuildDistributedModel(model=self.segmentor, model_cfg={})
        # load history checkpoints
        if self.history_segmentor is not None and mode == 'TRAIN':
            history_task_work_dir = os.path.join(runner_cfg['work_dir'], f'task_{runner_cfg["task_id"] - 1}')
            ckpts = loadckpts(os.path.join(history_task_work_dir, 'latest.pth'))
            self.segmentor.load_state_dict(ckpts['segmentor'], strict=False)
            if hasattr(self.segmentor.module, 'initaddedclassifier'):
                self.segmentor.module.initaddedclassifier(device=self.device)
            if hasattr(self, 'convertsegmentors'):
                self.convertsegmentors()
            self.history_segmentor.load_state_dict(ckpts['segmentor'], strict=True)
            for param in self.history_segmentor.parameters():
                param.requires_grad = False
            self.history_segmentor.eval()
        # load current checkpoints
        if os.path.islink(os.path.join(self.task_work_dir, 'latest.pth')) and mode == 'TRAIN':
            ckpts = loadckpts(os.path.join(self.task_work_dir, 'latest.pth'))
            self.segmentor.load_state_dict(ckpts['segmentor'], strict=True)
            self.optimizer.load_state_dict(ckpts['optimizer'])
            self.scheduler.setstate(state_dict=ckpts)
            amp.load_state_dict(ckpts['amp'])
            self.best_score = ckpts['best_score']
    '''start'''
    def start(self):
        if self.cmd_args.local_rank == 0:
            self.logger_handle.info(f'Load Config From: {self.cmd_args.cfgfilepath}')
            self.logger_handle.info(f'Config Details: \n{self.runner_cfg}')
        self.beforetrainactions()
        for cur_epoch in range(self.scheduler.cur_epoch+1, self.scheduler.max_epochs+1):
            if self.cmd_args.local_rank == 0:
                self.logger_handle.info(f'Start to train {self.runner_cfg["algorithm"]} at Task {self.runner_cfg["task_id"]}, Epoch {cur_epoch}')
            self.scheduler.cur_epoch = cur_epoch
            self.train(cur_epoch=cur_epoch)
            if ((cur_epoch % self.save_interval_epochs == 0) or (cur_epoch == self.scheduler.max_epochs)) and (self.cmd_args.local_rank == 0):
                ckpt_path = os.path.join(self.task_work_dir, f'epoch_{cur_epoch}.pth')
                saveckpts(ckpts=self.state(), savepath=ckpt_path)
                symlink(ckpt_path, os.path.join(self.task_work_dir, 'latest.pth'))
            if (cur_epoch % self.eval_interval_epochs == 0) or (cur_epoch == self.scheduler.max_epochs):
                results = self.test(cur_epoch=cur_epoch)
                if self.cmd_args.local_rank == 0:
                    ckpt_path = os.path.join(self.task_work_dir, f'epoch_{cur_epoch}.pth')
                    if self.best_score <= results[self.choose_best_segmentor_by_metric]:
                        self.best_score = results[self.choose_best_segmentor_by_metric]
                        symlink(ckpt_path, os.path.join(self.task_work_dir, 'best.pth'))
                        saveaspickle(results, os.path.join(self.task_work_dir, 'best.pkl'))
                    self.logger_handle.info(results)
        self.aftertrainactions()
        if self.cmd_args.local_rank == 0:
            best_results = loadpicklefile(os.path.join(self.task_work_dir, 'best.pkl'))
            self.logger_handle.info(f'Best Result at Task {self.runner_cfg["task_id"]}: \n{best_results}')
    '''aftertrainactions'''
    def aftertrainactions(self):
        pass
    '''beforetrainactions'''
    def beforetrainactions(self):
        pass
    '''train'''
    def train(self, cur_epoch):
        raise NotImplementedError('not to be implemented')
    '''test'''
    @torch.no_grad()
    def test(self, cur_epoch):
        if self.cmd_args.local_rank == 0:
            self.logger_handle.info(f'Start to test {self.runner_cfg["algorithm"]} at Task {self.runner_cfg["task_id"]}, Epoch {cur_epoch}')
        self.segmentor.eval()
        seg_evaluator = SegmentationEvaluator(num_classes=self.runner_cfg['num_total_classes'])
        with torch.no_grad():
            test_loader = self.test_loader
            if self.cmd_args.local_rank == 0:
                test_loader = tqdm(self.test_loader)
                test_loader.set_description('Evaluating')
            for batch_idx, data_meta in enumerate(test_loader):
                images = data_meta['image'].to(self.device, dtype=torch.float32)
                seg_targets = data_meta['seg_target'].to(self.device, dtype=torch.long)
                seg_logits = self.segmentor(images)['seg_logits']
                seg_logits = F.interpolate(seg_logits, size=seg_targets.shape[-2:], mode='bilinear', align_corners=self.segmentor.module.align_corners)
                seg_preds = seg_logits.max(dim=1)[-1]
                seg_targets = seg_targets.cpu().numpy()
                seg_preds = seg_preds.cpu().numpy()
                seg_evaluator.update(seg_targets=seg_targets, seg_preds=seg_preds)
        seg_evaluator.synchronize(device=self.device)
        results = seg_evaluator.evaluate()
        self.segmentor.train()
        return results
    '''state'''
    def state(self):
        state_dict = self.scheduler.state()
        state_dict.update({
            'best_score': self.best_score,
            'segmentor': self.segmentor.state_dict(),
            'task_id': self.runner_cfg['task_id'],
            'amp': amp.state_dict(),
        })
        return state_dict
    '''loggingtraininginfo'''
    def loggingtraininginfo(self, seg_losses_log_dict, losses_log_dict, init_losses_log_dict):
        for key, value in seg_losses_log_dict.items():
            if key in losses_log_dict:
                losses_log_dict[key].append(value)
            else:
                losses_log_dict[key] = [value]
        losses_log_dict.update({
            'algorithm': self.runner_cfg['algorithm'], 'task_name': self.runner_cfg['task_name'], 'task_id': self.runner_cfg['task_id'], 
            'encoder': self.runner_cfg['segmentor_cfg']['encoder_cfg']['type'], 'decoder': self.runner_cfg['segmentor_cfg']['decoder_cfg']['type'],
            'cur_epoch': self.scheduler.cur_epoch, 'max_epochs': self.scheduler.max_epochs, 'cur_iter': self.scheduler.cur_iter, 'max_iters': self.scheduler.max_iters,
            'lr': self.scheduler.lr,
        })
        if (self.scheduler.cur_iter % self.log_interval_iterations == 0) and (self.cmd_args.local_rank == 0):
            for key, value in losses_log_dict.copy().items():
                if isinstance(value, list):
                    losses_log_dict[key] = sum(value) / len(value)
            self.logger_handle.info(losses_log_dict)
            losses_log_dict = copy.deepcopy(init_losses_log_dict)
        return losses_log_dict