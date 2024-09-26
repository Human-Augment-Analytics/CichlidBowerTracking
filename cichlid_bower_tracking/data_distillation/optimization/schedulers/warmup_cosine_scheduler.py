from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

import math

class WarmupCosineScheduler(_LRScheduler):
    def __init__(self, optimizer: Optimizer, warmup_epochs: int, total_epochs: int, eta_min=0.0, last_epoch=-1):
        '''
        Initializes an instance of the WarmupCosineScheduler.

        Inputs:
            optimizer: the optimizer that the scheduler will use.
            warmup_epochs: the number of epochs that warmup will be performed over.
            total_epochs: the total number of epochs to be used in training.
            eta_min: the minimum learning rate after cosine annealing; defaults to 0.0.
            last_epoch: the epoch number tracker to be used by the scheduler; defaults to -1.
        '''

        super(WarmupCosineScheduler, self).__init__(optimizer)

        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min

    def get_lr(self) -> List[float]:
        '''
        Returns a learning rate via linear warmup scheduler with cosine annealing, as used to train the Pyramid Vision Transformer (PVT).

        Inputs: none.

        Returns: a list of learning rates.
        '''

        if self.last_epoch < self.warmup_epochs:
            return [(base_lr * self.last_epoch / self.warmup_epochs) for base_lr in self.base_lrs]
        else:
            return [self.eta_min + 0.5 * (base_lr - self.eta_min) * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs))) for base_lr in self.base_lrs]
        