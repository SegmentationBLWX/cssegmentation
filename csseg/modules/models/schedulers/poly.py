'''
Funtion:
    Implementation of PolyScheduler
Author:
    Zhenchao Jin
'''
from .base import BaseScheduler


'''PolyScheduler'''
class PolyScheduler(BaseScheduler):
    def __init__(self, power=0.9, optimizer=None, lr=0.01, min_lr=None, warmup_cfg=None, clipgrad_cfg=None, max_epochs=-1, iters_per_epoch=-1, paramwise_cfg=dict()):
        super(PolyScheduler, self).__init__(
            optimizer=optimizer, lr=lr, min_lr=min_lr, warmup_cfg=warmup_cfg, clipgrad_cfg=clipgrad_cfg, 
            max_epochs=max_epochs, iters_per_epoch=iters_per_epoch, paramwise_cfg=paramwise_cfg,
        )
        self.power = power
    '''updatelr'''
    def updatelr(self):
        # obtain variables
        base_lr, min_lr, cur_iter, max_iters, power = self.lr, self.min_lr, self.cur_iter, self.max_iters, self.power
        optimizer, warmup_cfg, paramwise_cfg = self.optimizer, self.warmup_cfg, self.paramwise_cfg
        # calculate target learning rate
        coeff = (1 - cur_iter / max_iters) ** power
        target_lr = coeff * (base_lr - min_lr) + min_lr
        if (warmup_cfg is not None) and (warmup_cfg['iters'] >= cur_iter):
            target_lr = self.getwarmuplr(cur_iter, warmup_cfg, target_lr)
        # update learning rate
        for param_group in optimizer.param_groups:
            if paramwise_cfg and paramwise_cfg.get('type', 'DefaultParamsConstructor') == 'DefaultParamsConstructor':
                param_group['lr'] = param_group.get('lr_multiplier', 1.0) * target_lr
            else:
                param_group['lr'] = target_lr
        # return
        return target_lr
    '''step'''
    def step(self):
        if self.clipgrad_cfg is not None:
            for param_group in self.optimizer.param_groups:
                self.clipgradients(params=param_group['params'], **self.clipgrad_cfg)
        self.optimizer.step()
        self.cur_iter += 1