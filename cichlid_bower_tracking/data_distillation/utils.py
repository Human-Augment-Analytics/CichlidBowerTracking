from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch

def train(dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module, optimizer: optim.Optimizer, epoch: int) -> bool:
    n_batches = len(dataloader)

    # ====================================================================================================================
    # TODO: implement the actual training loop
    # ====================================================================================================================
    
    return True