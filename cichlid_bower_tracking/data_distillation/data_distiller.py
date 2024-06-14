from typing import Tuple
import copy

from models.siamese_autoencoder import SiameseAutoencoder
from losses.siamese_loss import SiameseLoss

from data_distillation.misc.epoch_tracker import EpochTracker
from data_distillation.misc.epoch_logger import EpochLogger

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch

class DataDistiller:
    def __init__(self, dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module, optimizer: optim.Optimizer, nepochs: int, save_best_weights: bool, save_fp: str, device: str):
        '''
        Initializes an instance of the DataDistiller class.

        Inputs:
            dataloader: a PyTorch dataloader containing pairs of bbox images collected by the BBoxCollector, as well as associated similarity labels for each pair.
            model: a PyTorch model to be trained/validated and used in distilling the data in the passed dataloader.
            loss_fn: a PyTorch module representing the loss function to be used in evaluating the passed model during training/validation.
            optimizer: a PyTorch Optimizer to be used in updating the passed model's weights during training.
            nepochs: an integer indicating the number of epochs over which training/validation will be performed.
            save_best_weights: a Boolean indicating whether or not the autoencoder's best training/validation weights should be saved (overwrites any currently saved weights at the same passed filepath).
            save_fp: a string representing the local filepath where the model's weights will be stored (only effective if save_best_weights = True).
            device: a string indicating the device on which training/validation and distillation will occur (should be either "cpu" or "gpu").
        '''
        
        self.__version__ = '0.1.0'

        self.dataloader = dataloader
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.train_logger = EpochLogger(value_type='Training Loss')
        self.valid_logger = EpochLogger(value_type='Validation Loss')
        self.nepochs = nepochs

        self.save_best_weights = save_best_weights
        self.save_fp = save_fp

        self.device = device

    def _train(self, epoch: int) -> Tuple[float]:
        '''
        For a given epoch, performs batch training.

        Inputs:
            epoch: an integer representing the current epoch of training.

        Returns:
            epoch_tracker.min: the minimum loss stored in the epoch_tracker.
            epoch_tracker.max: the maximum loss stored in the epoch_tracker.
            epoch_tracker.avg: the average loss stored in the epoch_tracker.
        '''

        # get the total number of batches in the dataloader for more informative print messages.
        nbatches = len(self.dataloader)

        # loop over dataloader to perform training
        epoch_tracker = EpochTracker()
        for batch, (x1, x2, y) in enumerate(self.dataloader):
            # move to CUDA if requested (and able)
            if self.device == 'gpu' and torch.cuda.is_available():
                x1 = x1.cuda()
                x2 = x2.cuda()
                y = y.cuda()

            # depending on model type, expect different outputs from forward pass
            if isinstance(self.model, SiameseAutoencoder):
                z1, z2, x1_reconstruction, x2_reconstruction = self.model(x1, x2)
            else: # if/when necessary, add more model types in elif-statements
                raise Exception(f'Invalid model type: must be SiameseAutoencoder.')

            # calculate loss using passed loss_fn
            if isinstance(self.loss_fn, SiameseLoss):
                loss = self.loss_fn(y, x1, x2, z1, z2, x1_reconstruction, x2_reconstruction)
            else: # if/when necessary, add more loss functions in elif-statements
                raise Exception(f'Invalid loss type: must be SiameseLoss.')
            
            # loss value tracking with EpochTracker
            epoch_tracker.add(loss)

            # backward pass
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # print statements
            if batch % 10 == 0:
                print(f'Epoch: {epoch} [{batch}/{nbatches}]\t' + \
                      f'Loss: {loss:.4f} [{epoch_tracker.avg:.4f}]')

        # return the epoch statistics as tracked by the EpochTracker 
        return epoch_tracker.min, epoch_tracker.max, epoch_tracker.avg
    
    def _validate(self, epoch: int) -> Tuple[float]:
        '''
        For a given epoch, performs batch validation.

        Inputs:
            epoch: an integer representing the current epoch of validation.

        Returns:
            epoch_tracker.min: the minimum loss stored in the epoch_tracker.
            epoch_tracker.max: the maximum loss stored in the epoch_tracker.
            epoch_tracker.avg: the average loss stored in the epoch_tracker.
        '''

        # get the total number of batches and create an EpochTracker
        nbatches = len(self.dataloader)
        epoch_tracker = EpochTracker()

        # set model to evaluation mode, prevent model updates with torch.no_grad()
        self.model.eval()
        with torch.no_grad():
            # manually keep track of batch number while looping through dataloader
            batch = 0

            # loop over dataloader to perform validation
            for (x1, x2, y) in enumerate(self.dataloader):
                # move to CUDA if requested (and able)
                if self.device == 'gpu' and torch.cuda.is_available():
                    x1 = x1.cuda()
                    x2 = x2.cuda()
                    y = y.cuda()

                # depending on model type, expect different outputs from forward pass
                if isinstance(self.model, SiameseAutoencoder):
                    z1, z2, x1_reconstruction, x2_reconstruction = self.model(x1, x2)
                else: # if/when necessary, add more model types in elif-statements
                    raise Exception(f'Invalid model type: must be SiameseAutoencoder.')
                
                # calculate loss using passed loss_fn
                if isinstance(self.loss_fn, SiameseLoss):
                    loss = self.loss_fn(y, x1, x2, z1, z2, x1_reconstruction, x2_reconstruction)
                else: # if/when necessary, add more loss functions in elif-statements
                    raise Exception(f'Invalid loss type: must be SiameseLoss.')
                
                # loss value tracking with EpochTracker
                epoch_tracker.add(loss)

                # print statements
                if batch % 10 == 0:
                    print(f'Epoch: {epoch} [{batch}/{nbatches}]\t' + \
                          f'Loss: {loss:.4f} [{epoch_tracker.avg:.4f}]')

                # increment manual batch count    
                batch += 1

        # return the epoch statistics as tracked by the EpochTracker             
        return epoch_tracker.min, epoch_tracker.max, epoch_tracker.avg
    
    def _main_loop(self) -> nn.Module:
        '''
        Performs the main training/validation loop for self.model.

        Inputs: None.

        Returns:
            best_model: a PyTorch model representing a copy of the best model observed during validation.
        '''

        # move model to CUDA if requested (and able)
        if self.device == 'gpu' and torch.cuda.is_available():
            self.model = self.model.cuda()

        # keep track of best average validation loss and best model weights
        best_valid_avg = float('inf')
        best_model = None

        # loop through passed number of epochs
        for epoch in range(self.nepochs):
            # perform training on current epoch
            print(f'\n---------------------------------------------------------------------------------------------')
            print(f'TRAINING')
            print(f'---------------------------------------------------------------------------------------------\n')

            
            train_min, train_max, train_avg = self._train(epoch=epoch)
            self.train_logger.add(train_min, train_max, train_avg)

            # perform validation on current epoch
            print(f'\n---------------------------------------------------------------------------------------------')
            print(f'VALIDATION')
            print(f'---------------------------------------------------------------------------------------------\n')

            valid_min, valid_max, valid_avg = self._validate(epoch=epoch)
            self.valid_logger.add(valid_min, valid_max, valid_avg)

            # if current model is an improvement, save it and its average loss
            if valid_avg < best_valid_avg:
                best_valid_avg = valid_avg
                best_model = copy.deepcopy(self.model)

        # final print statement
        print(f'=============================================================================================')
        print(f'BEST VALIDATION MODEL LOSS: {best_valid_avg}')
        print(f'=============================================================================================\n')

        # return the best model
        return best_model
    
    def _save_model_weights(self, best_model: nn.Module) -> bool:
        '''
        Saves the weights of the best model to the self.save_fp filepath.

        Inputs:
            best_model: a PyTorch model representing a copy of the best model observed during validation.

        Returns: a Boolean indicating if the best model's weights were successfully stored.
        '''

        print('Attempting to save best model weights...')
        try:
            torch.save(best_model.state_dict(), self.save_fp)

            return True
        except Exception:
            return False
        
    def run_main_loop(self) -> None:
        '''
        Combines the _main_loop and _save_best_weights private functions into a single public location.

        Inputs: None.
        '''

        best_model = self._main_loop()

        if self.save_best_weights:
            save_flag = self._save_model_weights(best_model=best_model)

            if save_flag:
                print(f'\tSave successful!')
            else:
                print(f'\tSave failed!')