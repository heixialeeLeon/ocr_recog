import torch
import torch.nn as nn

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def set_learning_rate(optimizer,lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class warmup_controller(object):
    def __init__(self, optimizer, warmup_lr=0.000001, warmup_epoch =5):
        self.optimizer = optimizer
        self.backup_lr = get_learning_rate(optimizer)
        self.warmup_lr = warmup_lr
        self.warmup_epoch = warmup_epoch

    def step(self, epoch, optimizer):
        if epoch < self.warmup_epoch:
            set_learning_rate(optimizer,self.warmup_lr)
        elif epoch == self.warmup_epoch:
            set_learning_rate(optimizer, self.backup_lr)
