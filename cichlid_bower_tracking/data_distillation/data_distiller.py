from typing import Tuple, Union
import copy
import tqdm

from data_distillation.models.convolutional.siamese_autoencoder import SiameseAutoencoder
from data_distillation.models.convolutional.triplet_autoencoder import TripletAutoencoder

from data_distillation.models.transformer.autoencoders.siamese_vit_autoencoder import SiameseViTAutoencoder
from data_distillation.models.transformer.autoencoders.triplet_vit_autoencoder import TripletViTAutoencoder
from data_distillation.models.transformer.feature_extractors.triplet_cross_attention_vit import TripletCrossAttentionViT as TCAiT

from data_distillation.losses.pairwise_losses.total_siamese_loss import TotalSiameseLoss
from data_distillation.losses.triplet_losses.total_triplet_loss import TotalTripletLoss
from data_distillation.losses.triplet_losses.triplet_classification_loss import TripletClassificationLoss

from data_distillation.misc.epoch_tracker import EpochTracker
from data_distillation.misc.epoch_logger import EpochLogger

from data_distillation.testing.data.pairs import Pairs
from data_distillation.testing.data.triplets import Triplets
from data_distillation.testing.data.test_triplets import TestTriplets
from data_distillation.testing.metrics.accuracy import Accuracy

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch

class DataDistiller:
    def __init__(self, train_dataloader: DataLoader, valid_dataloader: DataLoader, model: Union[TCAiT, SiameseAutoencoder, TripletAutoencoder, SiameseViTAutoencoder, TripletViTAutoencoder], \
                 loss_fn: Union[TripletClassificationLoss, TotalTripletLoss, TotalSiameseLoss], optimizer: optim.Optimizer, nepochs: int, nclasses: int, save_best_weights: bool, save_fp: str, device: str, disable_progress_bar=False):
        '''
        Initializes an instance of the DataDistiller class.

        Inputs:
            dataloader: a PyTorch dataloader containing the data necessary for training/validation.
            model: a PyTorch model to be trained/validated and used in distilling the data in the passed dataloader.
            loss_fn: a PyTorch module representing the loss function to be used in evaluating the passed model during training/validation.
            optimizer: a PyTorch Optimizer to be used in updating the passed model's weights during training.
            nepochs: an integer indicating the number of epochs over which training/validation will be performed.
            save_best_weights: a Boolean indicating whether or not the autoencoder's best training/validation weights should be saved (overwrites any currently saved weights at the same passed filepath).
            save_fp: a string representing the local filepath where the model's weights will be stored (only effective if save_best_weights = True).
            device: a string indicating the device on which training/validation and distillation will occur (should be either "cpu" or "gpu").
            disable_progress_bar: a Boolean flag indicating whether or not a progress bar should be printed; defaults to False.
        '''
        
        self.__version__ = '0.3.0'

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.use_pairwise_data = isinstance(self.train_dataloader.dataset, Pairs)
        self.use_triplet_data = isinstance(self.train_dataloader.dataset, Triplets) or isinstance(self.train_dataloader.dataset, TestTriplets)

        self.train_logger = EpochLogger(value_type='Training Loss')
        self.valid_logger = EpochLogger(value_type='Validation Loss')

        self.acc_train_logger = EpochLogger(value_type='Training Accuracy')
        self.acc_valid_logger = EpochLogger(value_type='Valudation Accuracy')

        self.nepochs = nepochs
        self.nclasses = nclasses

        self.save_best_weights = save_best_weights
        self.save_fp = save_fp

        self.device = device
        self.disable_progress_bar = disable_progress_bar

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
        nbatches = len(self.train_dataloader)

        # condition on dataset type
        if self.use_pairwise_data:
            loop = tqdm.tqdm(self.train_dataloader, total=nbatches, position=0, disable=self.disable_progress_bar)

            # loop over dataloader to perform training
            epoch_tracker = EpochTracker()
            for batch, (x1, x2, y) in enumerate(loop):
                # move to CUDA if requested (and able)
                if self.device == 'gpu' and torch.cuda.is_available():
                    x1 = x1.cuda()
                    x2 = x2.cuda()
                    y = y.cuda()

                # depending on model type, expect different outputs from forward pass
                if isinstance(self.model, SiameseAutoencoder) or isinstance(self.model, SiameseViTAutoencoder):
                    z1, z2, x1_reconstruction, x2_reconstruction = self.model(x1, x2)
                else: # if/when necessary, add more model types in elif-statements
                    raise Exception(f'Invalid model type: must be SiameseAutoencoder or SiameseViTAutoencoder.')

                # calculate loss using passed loss_fn
                if isinstance(self.loss_fn, TotalSiameseLoss):
                    loss = self.loss_fn(y, x1, x2, z1, z2, x1_reconstruction, x2_reconstruction)
                else: # if/when necessary, add more loss functions in elif-statements
                    raise Exception(f'Invalid loss type: must be TotalSiameseLoss.')

                # backward pass
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # loss value tracking with EpochTracker
                epoch_tracker.add(loss.item())

                # update progress bar
                loop.set_description(f'Batch [{batch}/{nbatches}]')
                loop.set_postfix(loss=epoch_tracker.avg)

            # return the epoch statistics as tracked by the EpochTracker 
            return epoch_tracker.min, epoch_tracker.max, epoch_tracker.avg
        
        elif self.use_triplet_data:
            loop = tqdm.tqdm(self.train_dataloader, total=nbatches, position=0, disable=self.disable_progress_bar)

            # loop over dataloader to perform training
            epoch_tracker = EpochTracker()

            metric = Accuracy()
            acc_tracker = EpochTracker()

            for batch, (anchor, positive, negative, y) in enumerate(loop):
                # move to CUDA if requested (and able)
                if self.device == 'gpu' and torch.cuda.is_available():
                    anchor = anchor.cuda()
                    positive = positive.cuda()
                    negative = negative.cuda()
                    y = y.cuda()
                
                # depending on model type, expect different outputs from forward pass
                if isinstance(self.model, TCAiT):
                    z_anchor, z_positive, z_negative, Y = self.model(anchor, positive, negative)

                    y_prob = torch.softmax(Y, dim=1)
                    y_pred = y_prob.argmax(dim=1)
                elif isinstance(self.model, TripletAutoencoder) or isinstance(self.model, TripletViTAutoencoder):
                    z_anchor, z_positive, z_negative, anchor_reconstruction, positive_reconstruction, negative_reconstruction = self.model(anchor, positive, negative)
                else: # if/when necessary, add more model types in elif-statements
                    raise Exception(f'Invalid model type: must be TCAiT, TripletAutoencoder, or TripletViTAutoencoder.')

                # calculate loss using passed loss_fn
                if isinstance(self.loss_fn, TripletClassificationLoss):
                    loss = self.loss_fn(z_anchor, z_positive, z_negative, y_prob, y)
                    acc = metric(y_pred, y).item()

                    acc_tracker.add(acc)
                elif isinstance(self.loss_fn, TotalTripletLoss):
                    loss = self.loss_fn(anchor, positive, negative, z_anchor, z_positive, z_negative, anchor_reconstruction, positive_reconstruction, negative_reconstruction)
                else: # if/when necessary, add more loss functions in elif-statements
                    raise Exception(f'Invalid loss type: must be TotalTripletLoss or TripletClassificationLoss.')

                # backward pass
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # loss value tracking with EpochTracker
                epoch_tracker.add(loss.item())

                # update progress bar
                loop.set_description(f'Training, Batch [{batch}/{nbatches}]')
                if not isinstance(self.model, TCAiT):
                    loop.set_postfix(loss=epoch_tracker.avg)
                else:
                    loop.set_postfix(loss=epoch_tracker.avg, accuracy=acc_tracker.avg)

            # return the epoch statistics as tracked by the EpochTracker
            if not isinstance(self.model, TCAiT):
                return epoch_tracker.min, epoch_tracker.max, epoch_tracker.avg
            else:
                return epoch_tracker.min, epoch_tracker.max, epoch_tracker.avg, acc_tracker.min, acc_tracker.max, acc_tracker.avg
        else:
            raise Exception('Invalid dataloader: please use dataloader with either pairwise or triplet-structured dataset.')
    
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
        nbatches = len(self.valid_dataloader)

        # set model to evaluation mode, prevent model updates with torch.no_grad()
        self.model.eval()
        with torch.no_grad():
            if self.use_pairwise_data:
                loop = tqdm.tqdm(self.valid_dataloader, total=nbatches, position=1, disable=self.disable_progress_bar)

                # loop over dataloader to perform validation
                epoch_tracker = EpochTracker()

                for batch, (x1, x2, y) in enumerate(loop):
                    # move to CUDA if requested (and able)
                    if self.device == 'gpu' and torch.cuda.is_available():
                        x1 = x1.cuda()
                        x2 = x2.cuda()
                        y = y.cuda()

                    # depending on model type, expect different outputs from forward pass
                    if isinstance(self.model, SiameseAutoencoder):
                        z1, z2, x1_reconstruction, x2_reconstruction = self.model(x1, x2)
                    else: # if/when necessary, add more model types in elif-statements
                        raise Exception('Invalid model type: must be SiameseAutoencoder.')
                    
                    # calculate loss using passed loss_fn
                    if isinstance(self.loss_fn, TotalSiameseLoss):
                        loss = self.loss_fn(y, x1, x2, z1, z2, x1_reconstruction, x2_reconstruction)
                    else: # if/when necessary, add more loss functions in elif-statements
                        raise Exception('Invalid loss type: must be SiameseLoss.')
                    
                    # loss value tracking with EpochTracker
                    epoch_tracker.add(loss.item())

                    # update progress bar
                    loop.set_description(f'Validation, Batch [{batch}/{nbatches}]')
                    loop.set_postfix(loss=epoch_tracker.avg)

                # return the epoch statistics as tracked by the EpochTracker             
                return epoch_tracker.min, epoch_tracker.max, epoch_tracker.avg
            elif self.use_triplet_data:
                loop = tqdm.tqdm(self.valid_dataloader, total=nbatches, position=1, disable=self.disable_progress_bar)

                # loop over dataloader to perform training
                epoch_tracker = EpochTracker()

                metric = Accuracy()
                acc_tracker = EpochTracker()
                
                for batch, (anchor, positive, negative, y) in enumerate(loop):
                    # move to CUDA if requested (and able)
                    if self.device == 'gpu' and torch.cuda.is_available():
                        anchor = anchor.cuda()
                        positive = positive.cuda()
                        negative = negative.cuda()
                        y = y.cuda()
                    
                    # depending on model type, expect different outputs from forward pass
                    if isinstance(self.model, TCAiT):
                        z_anchor, z_positive, z_negative, Y = self.model(anchor, positive, negative)

                        y_prob = torch.softmax(Y, dim=1)
                        y_pred = y_prob.argmax(dim=1)
                    elif isinstance(self.model, TripletAutoencoder) or isinstance(self.model, TripletViTAutoencoder):
                        z_anchor, z_positive, z_negative, anchor_reconstruction, positive_reconstruction, negative_reconstruction = self.model(anchor, positive, negative)
                    else: # if/when necessary, add more model types in elif-statements
                        raise Exception('Invalid model type: must be TCAiT, TripletAutoencoder, or TripletViTAutoencoder.')

                    # calculate loss using passed loss_fn
                    if isinstance(self.loss_fn, TripletClassificationLoss):
                        loss = self.loss_fn(z_anchor, z_positive, z_negative, y_prob, y)
                        acc = metric(y_pred, y).item()

                        acc_tracker.add(acc)
                    elif isinstance(self.loss_fn, TotalTripletLoss):
                        loss = self.loss_fn(anchor, positive, negative, z_anchor, z_positive, z_negative, anchor_reconstruction, positive_reconstruction, negative_reconstruction)
                    else: # if/when necessary, add more loss functions in elif-statements
                        raise Exception('Invalid loss type: must be TotalTripletLoss or TripletClassificationLoss.')

                    # loss value tracking with EpochTracker
                    epoch_tracker.add(loss.item())

                    # update progress bar
                    loop.set_description(f'Validation, Batch [{batch}/{nbatches}]')
                    if not isinstance(self.model, TCAiT):
                        loop.set_postfix(loss=epoch_tracker.avg)
                    else:
                        loop.set_postfix(loss=epoch_tracker.avg, accuracy=acc_tracker.avg)

                # return the epoch statistics as tracked by the EpochTracker 
                if not isinstance(self.model, TCAiT):
                    return epoch_tracker.min, epoch_tracker.max, epoch_tracker.avg
                else:
                    return epoch_tracker.min, epoch_tracker.max, epoch_tracker.avg, acc_tracker.min, acc_tracker.max, acc_tracker.avg
            else:
                raise Exception('Invalid dataloader: please use dataloader with either pairwise or triplet-structured dataset.')
    
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
            print('\n' + '-' * 93)
            print(f'EPOCH [{epoch}/{self.nepochs}]')
            print('-' * 93)

            # perform training on current epoch
            if not isinstance(self.model, TCAiT):
                train_min, train_max, train_avg = self._train(epoch=epoch)
                self.train_logger.add(train_min, train_max, train_avg)
            else:
                train_min, train_max, train_avg, train_acc_min, train_acc_max, train_acc_avg = self._train(epoch=epoch)
                
                self.train_logger.add(train_min, train_max, train_avg)
                self.acc_train_logger.add(train_acc_min, train_acc_max, train_acc_avg)

            # perform validation on current epoch
            if not isinstance(self.model, TCAiT):
                valid_min, valid_max, valid_avg = self._validate(epoch=epoch)
                self.train_logger.add(valid_min, valid_max, valid_avg)
            else:
                valid_min, valid_max, valid_avg, valid_acc_min, valid_acc_max, valid_acc_avg = self._validate(epoch=epoch)
                
                self.valid_logger.add(valid_min, valid_max, valid_avg)
                self.acc_valid_logger.add(valid_acc_min, valid_acc_max, valid_acc_avg)

            # if current model is an improvement, save it and its average loss
            if not isinstance(self.model, TCAiT):
                if valid_avg < best_valid_avg:
                    best_valid_avg = valid_avg
                    best_model = copy.deepcopy(self.model)

            else:
                if valid_acc_avg < best_valid_avg:
                    best_valid_avg = valid_acc_avg
                    best_model = copy.deepcopy(self.model)
        
        # final print statement
        print('\n' + '=' * 93)
        print(f'BEST VALIDATION MODEL {"LOSS" if not isinstance(self.model, TCAiT) else "ACCURACY"}: {best_valid_avg:.4f}')
        print('=' * 93 + '\n')

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
                print('\tSave successful!')
            else:
                print('\tSave failed!')