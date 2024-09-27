from data_distillation.models.transformer.feature_extractors.triplet_cross_attention_vit import TripletCrossAttentionViT as TCAiT
from data_distillation.models.transformer.feature_extractors.tcait_extractor import TCAiTExtractor
from data_distillation.models.transformer.feature_extractors.pyramid.pyra_tcait import PyraTCAiT

from data_distillation.losses.triplet_losses.triplet_classification_loss import TripletClassificationLoss as TCLoss
from data_distillation.losses.triplet_losses.triplet_loss import TripletLoss

from data_distillation.optimization.schedulers.warmup_cosine_scheduler import WarmupCosineScheduler

from data_distillation.testing.data.test_triplets import TestTriplets
from data_distillation.testing.data.triplets import Triplets
from data_distillation.data_distiller import DataDistiller

from torch.utils.data.dataloader import DataLoader

import torch.optim as optim
import torch.nn as nn
import torch

from torchvision.transforms import Compose, Resize, RandomResizedCrop, CenterCrop, RandomHorizontalFlip, ColorJitter, Normalize, Lambda

from sklearn.model_selection import train_test_split
import pandas as pd

# ======================================================================
# ARGUMENTS
# ----------------------------------------------------------------------

# dataset paths
DATASET_PATH = '/home/hice1/cclark339/scratch/Data/ImageNet-1K/imagenet1k-triplets.csv'

# data splitting args
NUM_CLASSES = 5
RANDOM_STATE = 42
TEST_SIZE = 0.1
SHUFFLE = True

# data augmentation args
TRAIN_DIM = 224
VALID_DIM = 256

BRIGHTNESS = 0.4
CONTRAST = 0.4
SATURATION = 0.4
HUE = 0.1

NORM_MEANS = [0.485, 0.456, 0.406]
NORM_STDS = [0.229, 0.224, 0.225]

# setup args
BATCH_SIZE = 16
CHANNELS = 3
NUM_EPOCHS = 10
CHECKPOINTS_DIR = './data_distillation/models/testing_weights'
DEVICE = 'gpu'
GPU_ID = 0
START_EPOCH = 0
USE_DDP = False
DISABLE_PROGRESS_BAR = False

# model args
EMBED_DIMS = [64, 128, 320, 512]
HEAD_COUNTS = [1, 2, 5, 8]
MLP_RATIOS = [8, 8, 4, 4]
SR_RATIOS = [8, 4, 2, 1]
DEPTHS = [3, 3, 6, 3]
PATCH_SIZE = 4
NUM_STAGES = 4
DROPOUT = 0.1
USE_IMPROVED = True
CLASSIFICATION_INTENT = True

# optimizer and scheduler args
USE_LABEL_SMOOTHING = False
LEARNING_RATE = 1e-4
BETAS = [0.9, 0.999]
WEIGHT_DECAY = 2.5e-4
PATIENCE = 10
WARMUP_EPOCHS = 5
ETA_MIN = 0.0

# ======================================================================

df = pd.read_csv(DATASET_PATH)
df = df[df['target'] < NUM_CLASSES]

train_df, valid_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=SHUFFLE)
del df

train_transform = [
    Lambda(lambda img: img.repeat(3, 1, 1) if img.shape[0] == 1 else img),
    RandomResizedCrop(TRAIN_DIM),
    RandomHorizontalFlip(),
    ColorJitter(BRIGHTNESS, CONTRAST, SATURATION, HUE),
    Normalize(NORM_MEANS, NORM_STDS)
]

valid_transform = [
    Lambda(lambda img: img.repeat(3, 1, 1) if img.shape[0] == 1 else img),
    Resize(VALID_DIM),
    CenterCrop(TRAIN_DIM),
    Normalize(NORM_MEANS, NORM_STDS)
]

train_dataset = Triplets(train_df, transform=Compose(train_transform))
valid_dataset = Triplets(valid_df, transform=Compose(valid_transform))

train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE)

model = PyraTCAiT(embed_dims=EMBED_DIMS, head_counts=HEAD_COUNTS, mlp_ratios=MLP_RATIOS, sr_ratios=SR_RATIOS, depths=DEPTHS,
                  num_stages=NUM_STAGES, dropout=DROPOUT, first_patch_dim=PATCH_SIZE, in_channels=CHANNELS, in_dim=TRAIN_DIM, 
                  add_classifier=CLASSIFICATION_INTENT, use_improved=USE_IMPROVED, classification_intent=CLASSIFICATION_INTENT, num_classes=NUM_CLASSES)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=tuple(BETAS), weight_decay=WEIGHT_DECAY)
scheduler = WarmupCosineScheduler(optimizer=optimizer, warmup_epochs=WARMUP_EPOCHS, total_epochs=NUM_EPOCHS, eta_min=ETA_MIN)

loss_fn = TCLoss(use_label_smoothing=USE_LABEL_SMOOTHING)

distiller = DataDistiller(train_dataloader=train_dataloader, valid_dataloader=valid_dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler, nepochs=NUM_EPOCHS, 
                          nclasses=NUM_CLASSES, checkpoints_dir=CHECKPOINTS_DIR, device=DEVICE, gpu_id=GPU_ID, start_epoch=START_EPOCH, ddp=USE_DDP, disable_progress_bar=DISABLE_PROGRESS_BAR)

distiller.main_loop()

