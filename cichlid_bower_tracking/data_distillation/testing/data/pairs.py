from typing import Tuple

import pandas as pd
from torch.utils.data.dataset import Dataset
from torchvision.io import read_image

class Pairs(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        '''
        Initializes an instance of the Pairs PyTorch Dataset.

        Inputs:
            df: a pandas DataFrame containing pairs of image filepaths and the associated similarity labels.
            transform: a set of PyTorch transforms to be performed on every image.
        '''
        super(Pairs, self).__init__()

        self.__version__ = '0.1.3'

        self.df = df
        self.transform = transform

    def __len__(self) -> int:
        '''
        Gets the number of image pairs and associated similarity labels in the Dataset.

        Inputs: None.

        Returns: the number of bbox image pairs and associated similarity labels in the Dataset.
        '''

        return self.df.shape[0]

    def __getitem__(self, index: int) -> Tuple:
        '''
        Obtains the image pair and associated similarity label stored at the passed index in self.df, then transforms the images (assuming self.transform is not None) and returns the transformed images and associated similarity label.

        Inputs:
            index: an integer index to be used in obtaining an image pair and associated similarity label.

        Returns:
            x1: the first bbox image in the pair at the passed index.
            x2: the second bbox image in the pair at the passed index.
            label: the similarity label associated with x1 and x2.
        '''

        x1_path, x2_path, label = self.df.iloc[index]
        x1, x2 = read_image(x1_path).float(), read_image(x2_path).float()

        if self.transform:
            x1 = self.transform(x1)
            x2 = self.transform(x2)

        return (x1, x2), label