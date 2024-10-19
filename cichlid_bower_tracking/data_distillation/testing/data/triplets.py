from typing import Tuple

import pandas as pd
from torch.utils.data.dataset import Dataset
from torchvision.io import read_image

class Triplets(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        '''
        Initializes an instance of a Triplets PyTorch dataset.

        Inputs:
            df: a pandas DataFrame containing triples of image filepaths.
            transform: a set of PyTorch transforms to be performed on every image.
        '''
        super(Triplets, self).__init__()

        self.__version__ = '0.1.1'

        self.df = df
        self.transform = transform

    def __len__(self) -> int:
        '''
        Gets the number of image triplets in the Dataset.

        Inputs: None

        Returns: the number of image triplets in the Dataset.
        '''

        return self.df.shape[0]
    
    def __getitem__(self, index: int) -> Tuple:
        '''
        Obtains the image triplet stored at the passed index in self.df, then transforms the images (assuming self.transform is not None) and returns the transformed images.

        Inputs:
            index: an integer index to be used in obtaining an image pair and associated similarity label.

        Returns:
            anchor_id: the identity of the anchor image in the triplet at the passed index.
            positive_id: the identity of the positive image (similar to anchor) in the triplet at the passed index.
            negative_id: the identity of the negative image (dissimilar to anchor) in the triplet at the passed index.
            anchor_path: the filepath of the anchor image in the triplet at the passed index.
            positive_path: the filepath of the positive image (similar to anchor) in the triplet at the passed index.
            negative_path: the filepath of the negative image (dissimilar to anchor) in the triplet at the passed index.
            anchor: the anchor image in the triplet at the passed index.
            positive: the positive image (similar to anchor) in the triplet at the passed index.
            negative: the negative image (dissimilar to anchor) in the triplet at the passed index.
        '''

        anchor_id, positive_id, negative_id = anchor_path, positive_path, negative_path = self.df.iloc[index]
        anchor, positive, negative = read_image(anchor_path).float(), read_image(positive_path).float(), read_image(negative_path).float()

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        assert anchor.shape[0] == 3, f'Anchor @ {anchor_path} is not RGB!'
        assert positive.shape[0] == 3, f'Positive @ {positive_path} is not RGB!'
        assert negative.shape[0] == 3, f'Negative @ {negative_path} is not RGB!'

        return (anchor_id, positive_id, negative_id, anchor_path, positive_path, negative_path, anchor, positive, negative)