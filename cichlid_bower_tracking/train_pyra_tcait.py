from typing import Tuple
import argparse, time

from data_distillation.models.transformer.feature_extractors.triplet_cross_attention_vit import TripletCrossAttentionViT as TCAiT
from data_distillation.models.transformer.feature_extractors.tcait_extractor import TCAiTExtractor
from data_distillation.models.transformer.feature_extractors.pyramid.pyra_tcait import PyraTCAiT

from data_distillation.losses.triplet_losses.triplet_classification_loss import TripletClassificationLoss as TCLoss
from data_distillation.losses.triplet_losses.triplet_loss import TripletLoss

from data_distillation.optimization.schedulers.warmup_cosine_scheduler import WarmupCosineScheduler

from data_distillation.testing.data.test_triplets import TestTriplets
from data_distillation.testing.data.triplets import Triplets
from data_distillation.data_distiller import DataDistiller, ddp_setup

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import destroy_process_group

import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn as nn
import torch

import pandas as pd

# parse arguments
parser = argparse.ArgumentParser()

# setup arguments
# parser.add_argument('model', type=str, choices=['tcait', 'tcait-extractor', 'tcait-classifier', 'pyra-tcait'], help='The type of model to be used during training and validation.')
# parser.add_argument('checkpointsdir', type=str, help='The path to the directory where the checkpoint files will be stored.')
parser.add_argument('task', type=str, choices=['cls', 'reid'], help='The task for which the PyraT-CAiT model should be trained.')
parser.add_argument('scheduler', type=str, choices=['reduce-on-plateau', 'warmup-cosine'], help='The type of learning rate scheduler to use.')
parser.add_argument('trainset', type=str, help='The filepath to the dataset to be used in training.')
parser.add_argument('validset', type=str, help='The filepath to the dataset to be used in validation.')
parser.add_argument('--batch-size', '-B', type=int, default=16, help='The size of each batch to be used by both \"datasets\".')
# parser.add_argument('--num-train-examples', '-t', type=int, default=11060223, help='The number of examples to be used by the training \"dataset\".')
# parser.add_argument('--num-valid-examples', '-v', type=int, default=522500, help='The number of examples to be used by the validation \"dataset\".')
# parser.add_argument('--num-train-batches', '-T', type=int, default=691264, help='The number of batches to be used by the training \"dataset\"; meaningless if option \"--num-train-examples\"/\"-t\" < 1.')
# parser.add_argument('--num-valid-batches', '-V', type=int, default=23657, help='The number o batches to be used by the validation \"dataset\"; meaningless if option \"--num-train-examples\"/\"-t\" < 1.')
parser.add_argument('--num-classes', '-n', type=int, default=10450, help='The number of classes to be used in the classifier\'s head MLP; defaults to 10450.')
parser.add_argument('--channels', '-c', type=int, choices=[1, 3], default=3, help='The number of channels in the images (1 for greyscale, 3 for RGB); defaults to 3.')
parser.add_argument('--dim', '-b', type=int, default=224, help='The dimension of the images; defaults to 224.')
parser.add_argument('--use-ddp', '-U', default=False, action='store_true', help='Indicates whether training should be distributed across multiple GPUs.')
parser.add_argument('--num-epochs', '-E', type=int, default=1, help='The number of epochs to use in the time trial; defaults to 1.')
parser.add_argument('--start-epoch', '-e', type=int, default=0, help='The epoch to start training from; only useful if resuming from previous training; defaults to 0.')
parser.add_argument('--device-type', '-i', type=str, choices=['gpu', 'cpu'], default='gpu', help='The device type to use during training/validation; defaults to \'gpu\'.')                    
parser.add_argument('--num-workers', '-W', type=int, default=0, help='The number of workers to use in the dataloaders.')

# standard T-CAiT arguments
# parser.add_argument('--embed-dim', '-e', type=int, default=768, help='The embedding dimension to be used in the model; defaults to 768.')
# parser.add_argument('--num-extractor-heads', '-X', type=int, default=8, help='The number of attention heads to use in the extractor; defaults to 12.')
# parser.add_argument('--num-classifier-heads', '-C', type=int, default=8, help='The number of attention heads to use in the classifier; defaults to 12.')
# parser.add_argument('--extractor-depth', '-D', type=int, default=8, help='The number of transformer blocks to include in the extractor; defaults to 8.')
# parser.add_argument('--classifier-depth', '-d', type=int, default=4, help='The number of transformer blocks to include in the classifier; defaults to 4.')
# parser.add_argument('--extractor-dropout', '-Z', type=float, default=0.1, help='The dropout probability to be used in the extractor; defaults to 0.1.')
# parser.add_argument('--classifier-dropout', '-z', type=float, default=0.1, help='The dropout probability to be used in the classifier; defaults to 0.1.')
# parser.add_argument('--extractor-mlp-ratio', '-R', type=float, default=4.0, help='The size of the MLP hidden layer in each transformer block in the extractor, relative to the embedding size; defaults to 4.0.')
# parser.add_argument('--classifier-mlp-ratio', '-r', type=float, default=4.0, help='The size of the MLP hidden layer in each transformer block in the classifier, relative to the embedding size; defaults to 4.0.')
# parser.add_argument('--patch-kernel-size', '-k', type=int, default=3, help='The kernel size to be used in the mini-patch embedding (meaningless without using the \"--use-minipatch\"/\"-m\" option); defaults to 3.')
# parser.add_argument('--patch-stride', '-s', type=int, default=2, help='The stride to be used in the mini-patch embedding (meaningless without using the \"--use-minipatch\"/\"-m\" option); defaults to 2.')
# parser.add_argument('--patch-ratio', '-P', type=float, default=8.0, help='The rate at which the number of channels in the mini-patcher increases (meaningless without using the \"--use-minipatch\"/\"-m\" option); defaults to 8.0.')
# parser.add_argument('--patch-ratio-decay', '-x', type=float, default=0.5, help='The rate at which the \"--patch-ratio\"/\"-P\" value decays (meaningless without using the \"--use-minipatch\"/\"-m\" option); defaults to 0.5.')
# parser.add_argument('--patch-num-convs', '-N', type=int, default=5, help='The number of convolutions to use in the mini-patcher (meaningless without using the \"--use-minipatch\"/\"-m\" option); defaults to 5.')
# parser.add_argument('--use-minipatch', '-u', default=False, action='store_true', help='Indicates that the extractor should use a mini-patch embedding instead of a standard embedding.')

# PyraT-CAiT arguments
parser.add_argument('--embed-dims', '-q', type=int, nargs='+', default=[64, 128, 320, 512], help='The embedding dimensions to use in the stages of a PyraT-CAiT model.')
parser.add_argument('--head-counts', '-H', type=int, nargs='+', default=[1, 2, 5, 8], help='The number of attention heads to use in the stages of a PyraT-CAiT model.')
parser.add_argument('--mlp-ratios', '-m', type=int, nargs='+', default=[8, 8, 4, 4], help='The MLP expansion ratios to be used in the stages of a PyraT-CAiT model.')
parser.add_argument('--sr-ratios', '-M', type=int, nargs='+', default=[8, 4, 2, 1], help='The spatial reduction ratios to be used in the stages of a PyraT-CAiT model.')
parser.add_argument('--depths', '-a', type=int, nargs='+', default=[3, 8, 27, 3], help='The depth of each stage of a PyraT-CAiT model.')
parser.add_argument('--num-stages', '-g', type=int, default=4, help='The number of stages to use in a PyraT-CAiT model.')
parser.add_argument('--dropout', '-w', type=float, default=0.1, help='The dropout probability to be used in the stages of a PyraT-CAiT model.')
# parser.add_argument('--add-classifier', '-A', default=False, action='store_true', help='Indicates that an MLP head should be added to the PyraT-CAiT model.')
parser.add_argument('--use-improved', '-I', default=False, action='store_true', help='Indicates whether the PyraT-CAiT model should use the improved stage implementation.')

# shared arguments
parser.add_argument('--patch-size', '-p', type=int, default=4, help='The patch size to be used in patch embedding (meaningless if using the \"--use-minipatch\"/\"-m\" option and model is T-CAiT-based); defaults to 16.')

# optimizer and scheduler arguments
parser.add_argument('--learning-rate', type=float, default=1e-4, help='The initial learning rate to be used by the AdamW optimizer.')
parser.add_argument('--betas', type=float, nargs='+', default=[0.9, 0.999], help='The beta values to be used by the AdamW optimizer.')
parser.add_argument('--weight-decay', type=float, default=2.5e-4, help='The weight decay to be used by the AdamW optimizer (default value inspired by Loshchilov and Hutter\'s "Fixing Weight Decay Regularization in Adam", Figure 2).')
parser.add_argument('--patience', type=int, default=10, help='The number of epochs without improvement before the scheduler reduces the learning rate; only useful for ReduceLROnPlateau scheduler.')
parser.add_argument('--warmup-epochs', type=int, default=5, help='The number of warmup epochs to be used by the scheduler; only useful for WarmupCosineScheduler.')
parser.add_argument('--eta-min', type=float, default=0.0, help='The minimum learning rate after cosine annealing; only useful for WarmupCosineScheduler.')

# miscellaneous arguments
parser.add_argument('--debug', default=False, action='store_true', help='Puts the script in debug mode so it outputs logging messages.')
parser.add_argument('--disable-progress-bar', default=False, action='store_true', help='Indicates that no progress bar should be printed out during the training/validation processes.')

args = parser.parse_args()

# define main function
def main(gpu_id: int, world_size: int):
    # setup DDP (if desired)
    if args.use_ddp:
        if args.debug:
            print('Setting up DDP...')

        ddp_setup(rank=gpu_id, world_size=world_size)

    # create datasets
    if args.debug:
        print('Creating testing and validaton datasets...')

    # train_dataset = TestTriplets(num_examples=args.num_train_examples, batch_size=args.batch_size, num_batches=args.num_train_batches, 
    #                             num_channels=args.channels, dim=args.dim, num_classes=args.num_classes)
    # valid_dataset = TestTriplets(num_examples=args.num_valid_examples, batch_size=args.batch_size, num_batches=args.num_valid_batches,
    #                             num_channels=args.channels, dim=args.dim, num_classes=args.num_classes)
    
    train_df = pd.read_csv(args.trainset)
    valid_df = pd.read_csv(args.validset)

    train_dataset = Triplets(df=train_df)
    valid_dataset = Triplets(df=valid_df)
        
    # create dataloaders
    if args.debug:
        print('Creating training and validaton dataloaders...')

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,sampler=DistributedSampler(dataset=train_dataset) if args.use_ddp else None)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, sampler=DistributedSampler(dataset=valid_dataset) if args.use_ddp else None)

    # create PyraT-CAiT model
    if args.debug:
        print('Creating PyraT-CAiT model...')
    
    # if args.model != 'pyra-tcait': 
    #     model = TCAiT(embed_dim=args.embed_dim, num_classes=args.num_classes, num_extractor_heads=args.num_extractor_heads, num_classifier_heads=args.num_classifier_heads,
    #                   in_channels=args.channels, in_dim=args.dim, extractor_depth=args.extractor_depth, extractor_dropout=args.extractor_dropout, extractor_mlp_ratio=args.extractor_mlp_ratio,
    #                   extractor_patch_dim=args.patch_size, extractor_patch_kernel_size=args.patch_kernel_size, extractor_patch_stride=args.patch_stride, extractor_patch_ratio=args.patch_ratio,
    #                   extractor_patch_ratio_decay=args.patch_ratio_decay, extractor_patch_n_convs=args.patch_num_convs, extractor_use_minipatch=args.use_minipatch, classifier_depth=args.classifier_depth,
    #                   classifier_dropout=args.classifier_dropout, classifier_mlp_ratio=args.classifier_mlp_ratio)
    
    #     if args.model == 'tcait-extractor':
    #         model = model.extractor
    #     elif args.model == 'tcait-classifier':
    #         model.freeze_extractor()
    # else:
    model = PyraTCAiT(embed_dims=args.embed_dims, head_counts=args.head_counts, mlp_ratios=args.mlp_ratios, sr_ratios=args.sr_ratios, depths=args.depths,
                      num_stages=args.num_stages, dropout=args.dropout, first_patch_dim=args.patch_size, in_channels=args.channels, in_dim=args.dim, 
                      add_classifier=(args.task == 'cls'), use_improved=args.use_improved, classification_intent=(args.task == 'cls'), num_classes=args.num_classes)

    if args.debug:
        print(model)

    # create optimizer
    if args.debug:
        print('Creating optimizer...')
    
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, betas=tuple(args.betas), weight_decay=args.weight_decay)
    if args.scheduler == 'reduce-on-plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=args.patience)
    else:
        args.scheduler = WarmupCosineScheduler(optimizer=optimizer, warmup_epochs=args.warmup_epochs, total_epochs=args.num_epochs, eta_min=args.eta_min)

    # create loss function
    if args.debug:
        print('Creating loss function...')

    # if args.model == 'tcait' or (args.model == 'pyra-tcait' and args.add_classifier):
    #     loss_fn = TCLoss()
    # elif args.model == 'tcait-extractor'  or (args.model == 'pyra-tcait' and not args.add_classifier):
    #     loss_fn = TripletLoss()
    # else:
    #     loss_fn = nn.CrossEntropyLoss()
        
    if args.task == 'cls':
        loss_fn = TCLoss()
    else:
        loss_fn = TripletLoss()

    # create data distiller
    if args.debug:
        print('Creating data distiller...')

    distiller = DataDistiller(train_dataloader=train_dataloader, valid_dataloader=valid_dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler, nepochs=args.num_epochs,
                              nclasses=args.num_classes, checkpoints_dir=args.checkpointsdir, device=args.device_type, gpu_id=gpu_id, start_epoch=args.start_epoch, ddp=args.use_ddp, disable_progress_bar=args.disable_progress_bar)

    # perform training/validation
    if args.debug:
        print('Performing testing and validaton...')
    
    start_time = time.time()
    distiller.main_loop()

    end_time = time.time()
    time_diff = end_time - start_time

    print(f'\nTime Difference: {time_diff:.4f} s')

    # destroy DDP processes (if necessary)
    if args.use_ddp:
        if args.debug:
            print('Shutting down DDP...')

        destroy_process_group()

if __name__ == '__main__':
    if args.use_ddp:
        world_size = torch.cuda.device_count()
        mp.spawn(main, args=(world_size, ), nprocs=world_size)
    else:
        main(gpu_id=0, world_size=1)