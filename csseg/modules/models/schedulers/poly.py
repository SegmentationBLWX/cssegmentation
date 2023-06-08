'''
Function:
    Implementation of PolyScheduler
Author:
    Zhenchao Jin
'''
from .base import BaseScheduler


'''PolyScheduler'''
class PolyScheduler(BaseScheduler):
    def __init__(self, optimizer, max_iters, max_epochs, lr, min_lr=None, power=0.9):
        super(PolyScheduler, self).__init__(
            optimizer=optimizer, max_iters=max_iters, max_epochs=max_epochs, lr=lr, min_lr=min_lr
        )
        self.power = power
    '''step'''
    def step(self):
        self.optimizer.step()
        self.cur_iter += 1
        self.cur_lr = self.updatelr()
    '''updatelr'''
    def updatelr(self):
        # obtain variables
        cur_iter, max_iters, power, optimizer = self.cur_iter, self.max_iters, self.power, self.optimizer
        base_lr, min_lr = self.lr, self.min_lr
        # calculate updated_lr
        coeff = (1 - cur_iter / max_iters) ** power
        updated_lr = coeff * (base_lr - min_lr) + min_lr
        # update optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group.get('lr_scale', 1.0) * updated_lr
        # return
        return updated_lr