from typing import Tuple, Union, List, Dict
import copy, tqdm, os, json

from data_distillation.models.convolutional.siamese_autoencoder import SiameseAutoencoder
from data_distillation.models.convolutional.triplet_autoencoder import TripletAutoencoder

from data_distillation.models.transformer.autoencoders.siamese_vit_autoencoder import SiameseViTAutoencoder
from data_distillation.models.transformer.autoencoders.triplet_vit_autoencoder import TripletViTAutoencoder

from data_distillation.models.transformer.feature_extractors.tcait_extractor import TCAiTExtractor
from data_distillation.models.transformer.feature_extractors.triplet_cross_attention_vit import TripletCrossAttentionViT as TCAiT
from data_distillation.models.transformer.feature_extractors.pyramid.pyra_tcait import PyraTCAiT
from data_distillation.models.transformer.feature_extractors.pyramid.pyramid_vision_transformer import PyramidVisionTransformer as PVT

from data_distillation.losses.pairwise_losses.total_siamese_loss import TotalSiameseLoss
from data_distillation.losses.triplet_losses.triplet_loss import TripletLoss
from data_distillation.losses.triplet_losses.total_triplet_loss import TotalTripletLoss
from data_distillation.losses.triplet_losses.triplet_classification_loss import TripletClassificationLoss

from data_distillation.optimization.schedulers.warmup_cosine_scheduler import WarmupCosineScheduler

from data_distillation.misc.epoch_tracker import EpochTracker
from data_distillation.misc.epoch_logger import EpochLogger
from data_distillation.misc.intra_epoch_logger import IntraEpochLogger

from data_distillation.testing.data.pairs import Pairs
from data_distillation.testing.data.triplets import Triplets
from data_distillation.testing.data.test_triplets import TestTriplets
from data_distillation.testing.metrics.accuracy import Accuracy

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group

from torchvision.transforms import Compose
from torchvision.io import read_image

import torch.distributed as dist
import torch.optim as optim
import torch.nn as nn
import torch
import timm

from scipy.spatial.distance import euclidean
import numpy as np
import pandas as pd
import random

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
    def __init__(self, df: pd.DataFrame, transform: Compose, valid_dataloader: DataLoader, model: Union[TCAiT, SiameseAutoencoder, TripletAutoencoder, SiameseViTAutoencoder, TripletViTAutoencoder, TCAiTExtractor, PyraTCAiT, PVT], \
                 scheduler: Union[optim.lr_scheduler.ReduceLROnPlateau, WarmupCosineScheduler], loss_fn: Union[TripletClassificationLoss, TotalTripletLoss, TotalSiameseLoss, TripletLoss, nn.CrossEntropyLoss], optimizer: optim.Optimizer, nepochs: int, \
                 batch_size: int, nclasses: int, ntriplets: int, nworkers: int, thetas: List[float], pretr_model: str, checkpoints_dir: str, embeddings_dir: str, triplets_dir: str, device: str, gpu_id: int, start_epoch=0, ddp=False, disable_progress_bar=False, \
                 max_attempts=100, p_max=1.0, margin=1.0, remine_freq=5):
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
            start_epoch: an integer representing the first epoch of training (only useful for resuming previous training); defaults to 0.
            ddp: indicates that the model should be wrapped in a DDP object; defaults to False.
            disable_progress_bar: a Boolean flag indicating whether or not a progress bar should be printed; defaults to False.
        '''
        
        self.__version__ = '0.6.1'

        self.df = df
        self.transform = transform
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
        self.scheduler = scheduler

        self.use_pairwise_data = isinstance(self.train_dataloader.dataset, Pairs)
        self.use_triplet_data = isinstance(self.train_dataloader.dataset, Triplets) or isinstance(self.train_dataloader.dataset, TestTriplets)

        self.train_logger = EpochLogger(value_type='Training Loss')
        self.valid_logger = EpochLogger(value_type='Validation Loss')

        self.acc_train_logger = EpochLogger(value_type='Training Accuracy')
        self.acc_valid_logger = EpochLogger(value_type='Valudation Accuracy')

        self.train_intra_epoch_loggers = [IntraEpochLogger(value_type='Loss') for _ in range(nepochs)]
        self.valid_intra_epoch_loggers = [IntraEpochLogger(value_type='Loss') for _ in range(nepochs)]

        self.nepochs = nepochs
        self.start_epoch = start_epoch
        self.nclasses = nclasses

        self.batch_size = batch_size
        self.nworkers = nworkers

        self.checkpoints_dir = checkpoints_dir.rstrip('/ ')
        self.embeddings_dir = embeddings_dir.rstrip('/ ')
        self.triplets_dir = triplets_dir.rstrip('/ ')

        self.disable_progress_bar = disable_progress_bar

        self.__init_mining(num_triplets=ntriplets, thetas=thetas, pretr_model=pretr_model, max_attempts=max_attempts, p_max=p_max, margin=margin, remine_freq=remine_freq)

    def __init_mining(self, num_triplets: int, thetas: List[float], pretr_model: str, max_attempts: int, p_max: float, margin: float, remine_freq: int):
        '''
        Initializes all the triplet mining-specific components.

        Inputs:
            num_triplets: the (maximum) total number of triplets to mine during every run of self._mine.
            thetas: the performance metric value constraints; used by the substrategy rate scheduler.
            pretr_model: the pre-trained model to be used in generating initial embeddings (until p > thetas[0]).
            max_attempts: the maximum number of attempts to use when mining semi-hard and hard negative triplets.
            p_max: the maximum possible value of the performance metric of choice.
            margin: the triplet margin.
            remine_freq: the frequency at which remining occurs (once every x epochs).
        '''
        
        _valid_pretr_models = {
            'resnet152', 'vit_large_patch16_224_in21k', 'vit_large_patch14_224_clip_laion2b', 
            'tf_efficientnet_b5.ns_jft_in1k', 'tf_efficientnet_b6.ns_jft_in1k',
            'deit_base_distilled_patch16_224.fb_in1k', 'seresnext101d_32x8d.ah_in1k'
        }

        if pretr_model is not None:
            assert pretr_model in _valid_pretr_models, f'Invalid Pre-trained Model: must pick from {_valid_pretr_models} (got {pretr_model})'

        self.num_triplets = num_triplets
        self.thetas = thetas
        self.pretr_model = pretr_model
        self.max_attempts = max_attempts
        self.p_max = p_max
        self.margin = margin
        self.remine_freq = remine_freq

        if pretr_model == 'resnet152':
            self.pretr_model = timm.create_model(model_name='resnet152', pretrained=True, features_only=True)
        elif pretr_model == 'vit_large_patch16_224_in21k':
            self.pretr_model = timm.create_model(model_name='vit_large_patch16_224_in21k', pretrained=True, features_only=True)
        elif pretr_model == 'vit_large_patch14_224_clip_laion2b':
            self.pretr_model = timm.create_model(model_name='vit_large_patch14_224_clip_laion2b', pretrained=True, features_only=True)
        elif pretr_model == 'tf_efficientnet_b5.ns_jft_in1k':
            self.pretr_model = timm.create_model('tf_efficientnet_b5.ns_jft_in1k', pretrained=True, features_only=True)
        elif pretr_model == 'tf_efficientnet_b6.ns_jft_in1k':
            self.pretr_model = timm.create_model('tf_efficientnet_b6.ns_jft_in1k', pretrained=True, features_only=True)
        elif pretr_model == 'deit_base_distilled_patch16_224.fb_in1k':
            self.pretr_model = timm.create_model('deit_base_distilled_patch16_224.fb_in1k', pretrained=True, features_only=True)
        elif pretr_model == 'seresnext101d_32x8d.ah_in1k':
            self.pretr_model = timm.create_model('seresnext101d_32x8d.ah_in1k', pretrained=True, features_only=True)
        else:
            raise ValueError('Invalid Input for pretr_model argument.')
        
    def _initial_embed(self) -> None:
        '''
        Generates the initial embeddings store for triplet mining using self.df.

        Inputs: None
        '''

        self.embeddings = dict()

        unique_ids = self.df['identity'].unique()
        for unique_id in unique_ids:
            subset = self.df[self.df['identity'] == unique_id]

            for path in subset['path'].to_list():
                img = read_image(path).float()
                embed = self.pretr_model(img)[-1]

                self.embeddings[unique_id][path] = embed.item().flatten().to_list()

    def _omegas(self, p: float, p_max: float) -> Tuple[float]:
        '''
        Determines the triplet mining strategy's percentages (omega_1, omega_2, omega_3) based on performance thresholds (theta_1, theta_2) and the maximum possible performance.

        Inputs:
            p: the performance metric value.
            p_max: the maximum possible value of the performance metric.

        Returns:
            omega_1: the percentage of triplets to be selected using the random mining substrategy.
            omega_2: the percentage of triplets to be selected using the semi-hard negative mining substrategy.
            omega_3: the percentage of triplets to be selected using the hard negative mining strategy.
        '''

        omega_1 = 0.0
        omega_2 = 0.0
        omega_3 = 0.0

        theta_1, theta_2 = self.thetas

        if p < theta_1:
            omega_1 = 0.5
            omega_2 = 0.5
        elif theta_1 <= p < theta_2:
            p_scale = (p - theta_1) / (theta_2 - theta_1)

            omega_1 = 0.5 - 0.25 * p_scale
            omega_2 = 0.5 + 0.25 * p_scale
        else:
            p_scale = (p - theta_2) / (p_max - theta_2)

            omega_1 = 0.25 - 0.25 * p_scale
            omega_2 = 0.75 - 0.25 * p_scale
            omega_3 = 0.5 * p_scale

        return omega_1, omega_2, omega_3 

    def _mine(self, p: float, p_max: float, margin: float, epoch: int, max_attempts=100, transform=None) -> DataLoader:
        '''
        Performs triplet mining using self.embeddings.

        Inputs:
            p: the model's current performance metric value.
            p_max: the maximum performance metric value.
            margin: the triplet margin used in the stored loss function, used in semi-hard mining.
            epoch: the current epoch number.
            max_attempts: the maximum number of attempts for finding semi-hard and hard negatives, used in semi-hard and hard mining.
            transform: the augmentation transformations applied to each image.

        Returns:

        '''
        
        dataloader = None

        # calculate number of triplets for each mining substrategy
        pct_rand, pct_semihard, pct_hard = self._omegas(p, p_max)
        
        num_rand = int(pct_rand * self.num_triplets)
        num_semihard = int(pct_semihard * self.num_triplets)
        num_hard = int(pct_hard * self.num_triplets)

        rands = self._random_mine(num_rand)
        semihards = self._semihard_mine(num_semihard, margin=margin, max_attempts=max_attempts)
        hards = self._hard_mine(num_hard, max_attempts=max_attempts)

        triplets = rands + semihards + hards
        df = pd.DataFrame(triplets)

        self._save_triplets(df, epoch=epoch)

        dataset = Triplets(df, transform=transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.nworkers)

        return dataloader

    def _random_mine(self, num_triplets: int) -> List[Dict]:
        rands = []

        for _ in range(num_triplets):
            # select anchor
            anchor_id = random.choice(list(self.embeddings.keys()))
            anchor_path = random.choice(list(self.embeddings[anchor_id].keys()))

            # select positive
            positive_id = anchor_id
            positive_path = anchor_path
            while positive_path == anchor_path:
                positive_path = random.choice(list(self.embeddings[positive_id].keys()))

            # select negative
            negative_id = anchor_id
            while negative_id == anchor_id:
                negative_id = random.choice(list(self.embeddings.keys()))

            negative_path = random.choice(list(self.embeddings[negative_id].keys()))

            # create and append row
            row = {
                'anchor_id': anchor_id,
                'positive_id': positive_id,
                'negative_id': negative_id,
                'anchor_path': anchor_path,
                'positive_path': positive_path,
                'negative_path': negative_path
            }

            rands.append(row)

        print(f'{len(rands)} Randomly-Mined Triplets Selected!')

        return rands
    
    def _semihard_mine(self, num_triplets: int, margin: float, max_attempts=100) -> List[Dict]:
        semihards = []
        attempt = 0

        while len(semihards) < num_triplets and attempt < num_triplets * max_attempts:
            # select anchor
            anchor_id = random.choice(list(self.embeddings.keys()))
            anchor_path = random.choice(list(self.embeddings[anchor_id].keys()))
            anchor_embed = self.embeddings[anchor_id][anchor_path]

            # select positive
            positive_id = anchor_id
            positive_path = anchor_path
            while positive_path == anchor_path:
                positive_path = random.choice(list(self.embeddings[positive_id].keys()))

            positive_embed = self.embeddings[positive_id][positive_path]

            # select semi-hard negative
            found_semihard = False
            for _ in range(max_attempts):
                negative_id = anchor_id
                while negative_id == anchor_id:
                    negative_id = random.choice(list(self.embeddings.keys()))
                
                negative_path = random.choice(list(self.embeddings[negative_id].keys()))
                negative_embed = self.embeddings[negative_id][negative_path]

                if euclidean(anchor_embed, positive_embed) < euclidean(anchor_embed, negative_embed) < euclidean(anchor_embed, positive_embed) + margin:
                    found_semihard = True
                    break

            if found_semihard:
                row = {
                    'anchor_id': anchor_id,
                    'positive_id': positive_id,
                    'negative_id': negative_id,
                    'anchor_path': anchor_path,
                    'positive_path': positive_path,
                    'negative_path': negative_path
                }

                semihards.append(row)

            attempt += 1

        return semihards
    
    def _hard_mine(self, num_triplets: int, max_attempts=100) -> List[Dict]:
        hards = []
        attempt = 0

        while len(hards) < num_triplets and attempt < num_triplets * max_attempts:
            # select anchor
            anchor_id = random.choice(list(self.embeddings.keys()))
            anchor_path = random.choice(list(self.embeddings[anchor_id].keys()))
            anchor_embed = self.embeddings[anchor_id][anchor_path]

            # select positive
            positive_id = anchor_id
            positive_path = anchor_path
            while positive_path == anchor_path:
                positive_path = random.choice(list(self.embeddings[positive_id].keys()))

            positive_embed = self.embeddings[positive_id][positive_path]

            # select hard negative
            found_hard = False
            for _ in range(max_attempts):
                negative_id = anchor_id
                while negative_id == anchor_id:
                    negative_id = random.choice(list(self.embeddings.keys()))
                
                negative_path = random.choice(list(self.embeddings[negative_id].keys()))
                negative_embed = self.embeddings[negative_id][negative_path]

                if euclidean(anchor_embed, negative_embed) < euclidean(anchor_embed, positive_embed):
                    found_hard = True
                    break

            if found_hard:
                row = {
                    'anchor_id': anchor_id,
                    'positive_id': positive_id,
                    'negative_id': negative_id,
                    'anchor_path': anchor_path,
                    'positive_path': positive_path,
                    'negative_path': negative_path
                }

                hards.append(row)

            attempt += 1

        return hards

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
                    raise Exception('Invalid model type: must be SiameseAutoencoder or SiameseViTAutoencoder.')

                # calculate loss using passed loss_fn
                if isinstance(self.loss_fn, TotalSiameseLoss):
                    loss = self.loss_fn(y, x1, x2, z1, z2, x1_reconstruction, x2_reconstruction)
                else: # if/when necessary, add more loss functions in elif-statements
                    raise Exception('Invalid loss type: must be TotalSiameseLoss.')

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

            for batch, (anchor_id, positive_id, negative_id, anchor_path, positive_path, negative_path, anchor, positive, negative, y) in enumerate(loop):
                # move to CUDA if requested (and able)
                if self.device == 'gpu' and torch.cuda.is_available():
                    anchor = anchor.to(self.gpu_id)
                    positive = positive.to(self.gpu_id)
                    negative = negative.to(self.gpu_id)
                    y = y.to(self.gpu_id)
                
                # depending on model type, expect different outputs from forward pass
                if isinstance(self.model, TCAiT) or isinstance(self.model, PyraTCAiT) or isinstance(self.model, PVT):
                    z_anchor, z_positive, z_negative, Y = self.model(anchor, positive, negative)

                    self.embeddings[anchor_id][anchor_path] = z_anchor.item().flatten().to_list()
                    self.embeddings[positive_id][positive_path] = z_positive.item().flatten().to_list()
                    self.embeddings[negative_id][negative_path] = z_negative.item().flatten().to_list()

                    if Y is not None:
                        y_prob = torch.softmax(Y, dim=1)
                        y_pred = y_prob.argmax(dim=1)
                elif self.ddp and (isinstance(self.model.module, TCAiT) or isinstance(self.model.module, PyraTCAiT) or isinstance(self.model, PVT)):
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
                    raise Exception('Invalid model type: must be TCAiT, TCAiTExtractor, PyraTCAiT, PVT, TripletAutoencoder, or TripletViTAutoencoder.')

                # calculate loss using passed loss_fn
                if isinstance(self.loss_fn, TripletClassificationLoss):
                    loss, triplet_loss, ce_loss = self.loss_fn(z_anchor, z_positive, z_negative, y_prob, y)
                    acc = metric(y_pred, y).item()

                    self.train_intra_epoch_loggers[epoch].add(loss.item(), triplet_loss.item(), ce_loss.item())
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

                # backward pass
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # loss value tracking with EpochTracker
                epoch_tracker.add(loss.item())

                # update progress bar
                loop.set_description(f'Training, Batch [{batch}/{nbatches}]')
                if not isinstance(self.model, TCAiT) and not isinstance(self.model, PyraTCAiT) and not isinstance(self.model, PVT):
                    loop.set_postfix(loss=epoch_tracker.avg)
                else:
                    loop.set_postfix(loss=epoch_tracker.avg, accuracy=acc_tracker.avg)

            # return the epoch statistics as tracked by the EpochTracker
            if not isinstance(self.model, TCAiT) and not isinstance(self.model, PyraTCAiT) and not isinstance(self.model, PVT):
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
                
                for batch, (_, _, _, _, _, _, anchor, positive, negative, y) in enumerate(loop):
                    # move to CUDA if requested (and able)
                    if self.device == 'gpu' and torch.cuda.is_available():
                        anchor = anchor.to(self.gpu_id)
                        positive = positive.to(self.gpu_id)
                        negative = negative.to(self.gpu_id)
                        y = y.to(self.gpu_id)
                    
                    # depending on model type, expect different outputs from forward pass
                    if isinstance(self.model, TCAiT) or isinstance(self.model, PyraTCAiT) or isinstance(self.model, PVT):
                        z_anchor, z_positive, z_negative, Y = self.model(anchor, positive, negative)

                        if Y is not None:
                            y_prob = torch.softmax(Y, dim=1)
                            y_pred = y_prob.argmax(dim=1)
                    elif self.ddp and (isinstance(self.model.module, TCAiT) or isinstance(self.model.module, PyraTCAiT) or isinstance(self.model, PVT)):
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
                        raise Exception('Invalid model type: must be TCAiT, TCAiTExtractor, PyraTCAiT, PVT, TripletAutoencoder, or TripletViTAutoencoder.')
                    
                    # calculate loss using passed loss_fn
                    if isinstance(self.loss_fn, TripletClassificationLoss):
                        loss, triplet_loss, ce_loss = self.loss_fn(z_anchor, z_positive, z_negative, y_prob, y)
                        acc = metric(y_pred, y).item()

                        self.valid_intra_epoch_loggers[epoch].add(loss.item(), triplet_loss.item(), ce_loss.item())
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
                    if not isinstance(self.model, TCAiT) and not isinstance(self.model, PyraTCAiT) and not isinstance(self.model, PVT):
                        loop.set_postfix(loss=epoch_tracker.avg)
                    else:
                        loop.set_postfix(loss=epoch_tracker.avg, accuracy=acc_tracker.avg)

                # return the epoch statistics as tracked by the EpochTracker 
                if not isinstance(self.model, TCAiT) and not isinstance(self.model, PyraTCAiT) and not isinstance(self.model, PVT):
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
            'optim_state': self.optimizer.state_dict(),
            'sched_state': self.scheduler.state_dict()
        }

        checkpoint_path = self.checkpoints_dir + f'/checkpoint_epoch{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)

        if self.gpu_id == 0:
            print(f'Epoch {epoch} Checkpoint Saved!')

    def _save_embeddings(self, epoch: int, metric: float) -> None:
        '''
        Saves the current epoch's embeddings and performance metric value in a JSON file.

        Input:
            epoch: the current epoch number.
            metric: the current epoch's final average metric value.
        '''

        embeddings_file = {
            'store': self.embeddings,
            'epoch': epoch,
            'metric': metric
        }

        embeddings_path = self.embeddings_dir + f'/embeddings_epoch{epoch}.json'
        with open(embeddings_path, 'w') as file:
            json.dump(embeddings_file, file)

        if self.gpu_id == 0:
            print(f'Epoch {epoch} Embeddings Saved!')

    def _save_triplets(self, df: pd.DataFrame, epoch: int) -> None:
        '''
        Saves the current epoch's triplet dataset in a CSV file.

        Input:
            df: the current epoch's triplet dataset in a pandas DataFrame.
            epoch: the current epoch number.
        '''

        triplets_path = f'/triplets_epoch{epoch}.csv'

        df.to_csv(triplets_path)

        if self.gpu_id == 0:
            print(f'Epoch {epoch} Triplets Saved!')

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
            self.scheduler.load_state_dict(checkpoint['sched_state'])

            if self.gpu_id == 0:
                print('Previous Epoch\'s Checkpoint Loaded!')

    def _load_embeddings(self, epoch: int) -> float:
        '''
        Loads the previous epoch's embeddings and model performance from the JSON file they were stored in.

        Input:
            epoch: the current epoch number.

        Returns:
            metric: the performance metric value of the previous epoch.
        '''

        metric = 0.0
        if epoch > 0:
            embeddings_path = self.embeddings_dir + f'/embeddings_epoch{epoch - 1}.json'
        
            with open(embeddings_path, 'r') as file:
                tmp = json.load(file)['store']
                
                self.embeddings = tmp['store']
                metric = tmp['metric']

                del tmp

            if self.gpu_id == 0:
                print('Previous Epoch\'s Embeddings Loaded!')

        return metric
    
    def _load_triplets(self, epoch: int) -> pd.DataFrame:
        '''
        Loads the previous epoch's triplet dataset from the CSV file it was stored in.

        Inputs:
            epoch: the current epoch number.

        Returns:
            df: the triplet dataset in a pandas DataFrame (None if epoch <= 0).
        '''

        df = None
        if epoch > 0:
            triplets_path = f'/triplets_epoch{epoch}.csv'

            df = pd.read_csv(triplets_path)

        if self.gpu_id == 0:
            print('Previous Epoch\'s Triplets Loaded!')

        return df

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
        for epoch in range(self.start_epoch, self.nepochs):
            # load previous epoch's embeddings (and possibly triplets, maybe save triplets if necessary)
            if epoch == 0:
                self._initial_embed()
                self.train_dataloader = self._mine(0.0, self.p_max, self.margin, epoch, self.max_attempts, self.transform)
            else:
                p = self._load_embeddings(epoch)

                if epoch % self.remine_freq != 0:
                    df = self._load_triplets(epoch)
                    data = Triplets(df, self.transform)

                    self.train_dataloader = DataLoader(data, batch_size=self.batch_size, num_workers=self.nworkers)
                    self._save_triplets(df=df, epoch=epoch)
                else:
                    self.train_dataloader = self._mine(p, self.p_max, self.margin, epoch, self.max_attempts, self.transform)
                    self._save_triplets(df=self.train_dataloader.dataset.df, epoch=epoch)
            
            # load previous epoch's checkpoint
            self._load_checkpoint(epoch)

            # epoch start print statement
            if not (self.disable_progress_bar or (self.ddp and self.gpu_id != 0)):
                print('\n' + '-' * 93)
                print(f'EPOCH [{epoch}/{self.nepochs}]')
                print('-' * 93)

            # perform training on current epoch
            p = 0.0
            if not isinstance(self.model, TCAiT) and not isinstance(self.model, PyraTCAiT) and not isinstance(self.model, PVT):
                train_min, train_max, train_avg = self._train(epoch=epoch)
                self.train_logger.add(train_min, train_max, train_avg)
            else:
                train_min, train_max, train_avg, train_acc_min, train_acc_max, train_acc_avg = self._train(epoch=epoch)
                p = train_acc_avg

                self.train_logger.add(train_min, train_max, train_avg)
                self.acc_train_logger.add(train_acc_min, train_acc_max, train_acc_avg)

            # save embeddings
            if (self.ddp and self.device == 'gpu' and self.gpu_id == 0) or not self.ddp:
                self._save_embeddings(epoch=epoch, metric=p)

            # perform validation on current epoch
            if not isinstance(self.model, TCAiT) and not isinstance(self.model, PyraTCAiT) and not isinstance(self.model, PVT):
                valid_min, valid_max, valid_avg = self._validate(epoch=epoch)
                self.valid_logger.add(valid_min, valid_max, valid_avg)
            else:
                valid_min, valid_max, valid_avg, valid_acc_min, valid_acc_max, valid_acc_avg = self._validate(epoch=epoch)
                
                self.valid_logger.add(valid_min, valid_max, valid_avg)
                self.acc_valid_logger.add(valid_acc_min, valid_acc_max, valid_acc_avg)

            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(valid_avg)
            else:
                self.scheduler.step()

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