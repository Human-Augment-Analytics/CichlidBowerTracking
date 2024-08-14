from typing import Tuple, Union
import copy, tqdm, os

from data_distillation.models.convolutional.siamese_autoencoder import SiameseAutoencoder
from data_distillation.models.convolutional.triplet_autoencoder import TripletAutoencoder

from data_distillation.models.transformer.autoencoders.siamese_vit_autoencoder import SiameseViTAutoencoder
from data_distillation.models.transformer.autoencoders.triplet_vit_autoencoder import TripletViTAutoencoder

from data_distillation.models.transformer.feature_extractors.tcait_extractor import TCAiTExtractor
from data_distillation.models.transformer.feature_extractors.triplet_cross_attention_vit import TripletCrossAttentionViT as TCAiT
from data_distillation.models.transformer.feature_extractors.pyramid.pyra_tcait import PyraTCAiT

from data_distillation.losses.pairwise_losses.total_siamese_loss import TotalSiameseLoss
from data_distillation.losses.triplet_losses.triplet_loss import TripletLoss
from data_distillation.losses.triplet_losses.total_triplet_loss import TotalTripletLoss
from data_distillation.losses.triplet_losses.triplet_classification_loss import TripletClassificationLoss

from data_distillation.misc.epoch_tracker import EpochTracker
from data_distillation.misc.epoch_logger import EpochLogger

from data_distillation.testing.data.pairs import Pairs
from data_distillation.testing.data.triplets import Triplets
from data_distillation.testing.data.test_triplets import TestTriplets
from data_distillation.testing.metrics.accuracy import Accuracy

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group

import torch.distributed as dist
import torch.optim as optim
import torch.nn as nn
import torch

def ddp_setup(rank, world_size):
    '''
    Sets up Distributed Data Parallel (DDP)

    Inputs:
        rank: unique identifier.
        world_size: total number of processes.
    '''

    torch.cuda.set_device(rank)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    init_process_group(backend='nccl', rank=rank, world_size=world_size)

class DataDistiller:
    def __init__(self, train_dataloader: DataLoader, valid_dataloader: DataLoader, model: Union[TCAiT, SiameseAutoencoder, TripletAutoencoder, SiameseViTAutoencoder, TripletViTAutoencoder, TCAiTExtractor, PyraTCAiT], \
                 loss_fn: Union[TripletClassificationLoss, TotalTripletLoss, TotalSiameseLoss, TripletLoss, nn.CrossEntropyLoss], optimizer: optim.Optimizer, nepochs: int, nclasses: int, \
                 checkpoints_dir: str, device: str, gpu_id: int, ddp=False, disable_progress_bar=False):
        '''
        Initializes an instance of the DataDistiller class.

        Inputs:
            dataloader: a PyTorch dataloader containing the data necessary for training/validation.
            model: a PyTorch model to be trained/validated and used in distilling the data in the passed dataloader.
            loss_fn: a PyTorch module representing the loss function to be used in evaluating the passed model during training/validation.
            optimizer: a PyTorch Optimizer to be used in updating the passed model's weights during training.
            nepochs: an integer indicating the number of epochs over which training/validation will be performed.
            checkpoints_dir: a string representing the local directory path where checkpoints will be saved.
            device: a string indicating the device on which training/validation and distillation will occur (must be either "cpu" or "gpu").
            gpu_id: a string representing the gpu_id to be used.
            ddp: indicates that the model should be wrapped in a DDP object.
            disable_progress_bar: a Boolean flag indicating whether or not a progress bar should be printed; defaults to False.
        '''
        
        self.__version__ = '0.5.1'

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        assert device in ['gpu', 'cpu'], f'Invalid device: needed \"cpu\" or \"gpu\", got \"{device}\".'
        
        self.device = device
        self.ddp = ddp
        self.gpu_id = gpu_id

        self.model = model
        if self.device == 'gpu' and torch.cuda.is_available() and not self.ddp:
            self.model = self.model.to(self.gpu_id)
        elif self.device == 'gpu' and torch.cuda.is_available() and self.ddp:
            self.model = self.model.to(self.gpu_id)
            self.model = DDP(self.model, device_ids=[self.gpu_id])

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

        self.checkpoints_dir = checkpoints_dir.rstrip('/ ')
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
            loop = tqdm.tqdm(self.train_dataloader, total=nbatches, disable=(self.disable_progress_bar or (self.ddp and self.gpu_id != 0)))

            # loop over dataloader to perform training
            epoch_tracker = EpochTracker()
            for batch, (x1, x2, y) in enumerate(loop):
                # move to CUDA if requested (and able)
                if self.device == 'gpu' and torch.cuda.is_available():
                    x1 = x1.to(self.gpu_id)
                    x2 = x2.to(self.gpu_id)
                    y = y.to(self.gpu_id)

                # depending on model type, expect different outputs from forward pass
                if isinstance(self.model, SiameseAutoencoder) or isinstance(self.model, SiameseViTAutoencoder):
                    z1, z2, x1_reconstruction, x2_reconstruction = self.model(x1, x2)
                elif self.ddp and (isinstance(self.model.module, SiameseAutoencoder) or isinstance(self.model.module, SiameseViTAutoencoder)):
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
            loop = tqdm.tqdm(self.train_dataloader, total=nbatches, disable=(self.disable_progress_bar or (self.ddp and self.gpu_id != 0)))

            # loop over dataloader to perform training
            epoch_tracker = EpochTracker()

            metric = Accuracy()
            acc_tracker = EpochTracker()

            for batch, (anchor, positive, negative, y) in enumerate(loop):
                # move to CUDA if requested (and able)
                if self.device == 'gpu' and torch.cuda.is_available():
                    anchor = anchor.to(self.gpu_id)
                    positive = positive.to(self.gpu_id)
                    negative = negative.to(self.gpu_id)
                    y = y.to(self.gpu_id)
                
                # depending on model type, expect different outputs from forward pass
                if isinstance(self.model, TCAiT) or isinstance(self.model, PyraTCAiT):
                    z_anchor, z_positive, z_negative, Y = self.model(anchor, positive, negative)

                    if Y is not None:
                        y_prob = torch.softmax(Y, dim=1)
                        y_pred = y_prob.argmax(dim=1)
                elif self.ddp and (isinstance(self.model.module, TCAiT) or isinstance(self.model.module, PyraTCAiT)):
                    z_anchor, z_positive, z_negative, Y = self.model(anchor, positive, negative)

                    if Y is not None:
                        y_prob = torch.softmax(Y, dim=1)
                        y_pred = y_prob.argmax(dim=1)
                elif isinstance(self.model, TCAiTExtractor):
                    z_anchor, z_positive, z_negative = self.model(anchor, positive, negative)
                elif self.ddp and isinstance(self.model.module, TCAiTExtractor):
                    z_anchor, z_positive, z_negative = self.model(anchor, positive, negative)
                elif isinstance(self.model, TripletAutoencoder) or isinstance(self.model, TripletViTAutoencoder):
                    z_anchor, z_positive, z_negative, anchor_reconstruction, positive_reconstruction, negative_reconstruction = self.model(anchor, positive, negative)
                elif self.ddp and (isinstance(self.model.module, TripletAutoencoder) or isinstance(self.model.module, TripletViTAutoencoder)):
                    z_anchor, z_positive, z_negative, anchor_reconstruction, positive_reconstruction, negative_reconstruction = self.model(anchor, positive, negative)
                else: # if/when necessary, add more model types in elif-statements
                    raise Exception(f'Invalid model type: must be TCAiT, TCAiTExtractor, PyraTCAiT, TripletAutoencoder, or TripletViTAutoencoder.')

                # calculate loss using passed loss_fn
                if isinstance(self.loss_fn, TripletClassificationLoss):
                    loss = self.loss_fn(z_anchor, z_positive, z_negative, y_prob, y)
                    acc = metric(y_pred, y).item()

                    acc_tracker.add(acc)
                elif isinstance(self.loss_fn, TripletLoss):
                    loss = self.loss_fn(z_anchor, z_positive, z_negative)
                elif isinstance(self.loss_fn, nn.CrossEntropyLoss):
                    loss = self.loss_fn(y_prob, y)
                    acc = metric(y_pred, y).item()

                    acc_tracker.add(acc)
                elif isinstance(self.loss_fn, TotalTripletLoss):
                    loss = self.loss_fn(anchor, positive, negative, z_anchor, z_positive, z_negative, anchor_reconstruction, positive_reconstruction, negative_reconstruction)
                else: # if/when necessary, add more loss functions in elif-statements
                    raise Exception(f'Invalid loss type: must be TotalTripletLoss, TripletClassificationLoss, TripletLoss, or CrossEntropyLoss.')

                # backward pass
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # loss value tracking with EpochTracker
                epoch_tracker.add(loss.item())

                # update progress bar
                loop.set_description(f'Training, Batch [{batch}/{nbatches}]')
                if not isinstance(self.model, TCAiT) and not isinstance(self.model, PyraTCAiT):
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
                loop = tqdm.tqdm(self.valid_dataloader, total=nbatches, disable=(self.disable_progress_bar or (self.ddp and self.gpu_id != 0)))

                # loop over dataloader to perform validation
                epoch_tracker = EpochTracker()

                for batch, (x1, x2, y) in enumerate(loop):
                    # move to CUDA if requested (and able)
                    if self.device == 'gpu' and torch.cuda.is_available():
                        x1 = x1.to(self.gpu_id)
                        x2 = x2.to(self.gpu_id)
                        y = y.to(self.gpu_id)

                    # depending on model type, expect different outputs from forward pass
                    if isinstance(self.model, SiameseAutoencoder) or isinstance(self.model, SiameseViTAutoencoder):
                        z1, z2, x1_reconstruction, x2_reconstruction = self.model(x1, x2)
                    elif self.ddp and (isinstance(self.model.module, SiameseAutoencoder) or isinstance(self.model.module, SiameseViTAutoencoder)):
                        z1, z2, x1_reconstruction, x2_reconstruction = self.model(x1, x2)
                    else: # if/when necessary, add more model types in elif-statements
                        raise Exception('Invalid model type: must be SiameseAutoencoder or SiameseViTAutoencoder.')
                    
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
                loop = tqdm.tqdm(self.valid_dataloader, total=nbatches, disable=(self.disable_progress_bar or (self.ddp and self.gpu_id != 0)))

                # loop over dataloader to perform training
                epoch_tracker = EpochTracker()

                metric = Accuracy()
                acc_tracker = EpochTracker()
                
                for batch, (anchor, positive, negative, y) in enumerate(loop):
                    # move to CUDA if requested (and able)
                    if self.device == 'gpu' and torch.cuda.is_available():
                        anchor = anchor.to(self.gpu_id)
                        positive = positive.to(self.gpu_id)
                        negative = negative.to(self.gpu_id)
                        y = y.to(self.gpu_id)
                    
                    # depending on model type, expect different outputs from forward pass
                    if isinstance(self.model, TCAiT) or isinstance(self.model, PyraTCAiT):
                        z_anchor, z_positive, z_negative, Y = self.model(anchor, positive, negative)

                        if Y is not None:
                            y_prob = torch.softmax(Y, dim=1)
                            y_pred = y_prob.argmax(dim=1)
                    elif self.ddp and (isinstance(self.model.module, TCAiT) or isinstance(self.model.module, PyraTCAiT)):
                        z_anchor, z_positive, z_negative, Y = self.model(anchor, positive, negative)

                        if Y is not None:
                            y_prob = torch.softmax(Y, dim=1)
                            y_pred = y_prob.argmax(dim=1)
                    elif isinstance(self.model, TCAiTExtractor):
                        z_anchor, z_positive, z_negative = self.model(anchor, positive, negative)
                    elif self.ddp and isinstance(self.model.module, TCAiTExtractor):
                        z_anchor, z_positive, z_negative = self.model(anchor, positive, negative)
                    elif isinstance(self.model, TripletAutoencoder) or isinstance(self.model, TripletViTAutoencoder):
                        z_anchor, z_positive, z_negative, anchor_reconstruction, positive_reconstruction, negative_reconstruction = self.model(anchor, positive, negative)
                    elif self.ddp and (isinstance(self.model.module, TripletAutoencoder) or isinstance(self.model.module, TripletViTAutoencoder)):
                        z_anchor, z_positive, z_negative, anchor_reconstruction, positive_reconstruction, negative_reconstruction = self.model(anchor, positive, negative)
                    else: # if/when necessary, add more model types in elif-statements
                        raise Exception(f'Invalid model type: must be TCAiT, TCAiTExtractor, PyraTCAiT, TripletAutoencoder, or TripletViTAutoencoder.')
                    
                    # calculate loss using passed loss_fn
                    if isinstance(self.loss_fn, TripletClassificationLoss):
                        loss = self.loss_fn(z_anchor, z_positive, z_negative, y_prob, y)
                        acc = metric(y_pred, y).item()

                        acc_tracker.add(acc)
                    elif isinstance(self.loss_fn, TripletLoss):
                        loss = self.loss_fn(z_anchor, z_positive, z_negative)
                    elif isinstance(self.loss_fn, nn.CrossEntropyLoss):
                        loss = self.loss_fn(y_prob, y)
                        acc = metric(y_pred, y).item()

                        acc_tracker.add(acc)
                    elif isinstance(self.loss_fn, TotalTripletLoss):
                        loss = self.loss_fn(anchor, positive, negative, z_anchor, z_positive, z_negative, anchor_reconstruction, positive_reconstruction, negative_reconstruction)
                    else: # if/when necessary, add more loss functions in elif-statements
                        raise Exception('Invalid loss type: must be TotalTripletLoss, TripletClassificationLoss, TripletLoss, or CrossEntropyLoss.')

                    # loss value tracking with EpochTracker
                    epoch_tracker.add(loss.item())

                    # update progress bar
                    loop.set_description(f'Validation, Batch [{batch}/{nbatches}]')
                    if not isinstance(self.model, TCAiT) and not isinstance(self.model, PyraTCAiT):
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
    
    def _save_checkpoint(self, epoch: int) -> None:
        '''
        Saves the current epoch's model and optimizer state dictionaries in a checkpoint file.

        Input:
            epoch: the current epoch number.
        '''

        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optim_state': self.model.state_dict()
        }

        checkpoint_path = self.checkpoints_dir + f'/checkpoint_epoch{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)

        print(f'Epoch {epoch} Checkpoint Saved!')

    def _load_checkpoint(self, epoch: int) -> None:
        '''
        Loads the previous epoch's model and optimizer state dictionaries from its checkpoint file into the current model and optimizer.

        Inputs:
            epoch: the curret epoch number.
        '''

        if epoch > 0:
            checkpoint_path = self.checkpoints_dir + f'/checkpoint_epoch{epoch - 1}.pth'
            assert os.path.exists(checkpoint_path), f'Loading Error: Checkpoint file "{checkpoint_path}" does not exist.'

            if self.device == 'gpu':
                checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda'))
                if self.ddp:
                    self.model.module.load_state_dict(checkpoint['model_state'])
                else:
                    self.model.load_state_dict(checkpoint['model_state'])
            else:
                checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
                self.model.load_state_dict(checkpoint['model_state'])

            self.optimizer.load_state_dict(checkpoint['optim_state'])

            if self.gpu_id == 0:
                print('Previous Epoch\'s Checkpoint Loaded!')
    
    # def _save_model_weights(self, best_model: nn.Module) -> bool:
    #     '''
    #     Saves the weights of the best model to the self.save_fp filepath.

    #     Inputs:
    #         best_model: a PyTorch model representing a copy of the best model observed during validation.

    #     Returns: a Boolean indicating if the best model's weights were successfully stored.
    #     '''

    #     print('Attempting to save best model weights...')
    #     try:
    #         torch.save(best_model.state_dict(), self.save_fp)

    #         return True
    #     except Exception:
    #         return False

    def main_loop(self) -> None:
        '''
        Performs the main training/validation loop for self.model.

        Inputs: None.

        Returns:
            best_model: a PyTorch model representing a copy of the best model observed during validation.
        '''
        
        # loop through passed number of epochs
        for epoch in range(self.nepochs):
            # load previous epoch's checkpoint
            self._load_checkpoint(epoch)

            # epoch start print statement
            if not (self.disable_progress_bar or (self.ddp and self.gpu_id != 0)):
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

            # save checkpoint
            if (self.ddp and self.device == 'gpu' and self.gpu_id == 0) or not self.ddp:
                self._save_checkpoint(epoch)

            # synchronize processes
            if self.ddp:
                dist.barrier()
        
    # def run_main_loop(self) -> None:
    #     '''
    #     Combines the _main_loop and _save_best_weights private functions into a single public location.

    #     Inputs: None.
    #     '''

    #     best_model = self._main_loop()

    #     if self.save_best_weights:
    #         if not self.ddp:
    #             save_flag = self._save_model_weights(best_model=best_model)
    #         else:
    #             save_flag = self._save_model_weights(best_model=best_model.module)

    #         if save_flag:
    #             print('\tSave successful!')
    #         else:
    #             print('\tSave failed!')