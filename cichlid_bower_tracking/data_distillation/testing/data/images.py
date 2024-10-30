from typing import Tuple

import pandas as pd
from torch.utils.data.dataset import Dataset
from torchvision.io import read_image

class Images(Dataset):
    '''
    Only used for initial embedding in the DataDistiller class.
    '''

    def __init__(self, df: pd.DataFrame, base_dir: str, transform=None):
        '''
        Initializes a simple Images dataset, using the passed pandas DataFrame and base directory.

        Inputs:
            df: a pandas DataFrame containing identities and image paths.
            base_dir: the string base directory where the image dataset is stored.
            transform: the image transformation/augmentation sequence to be used after retrieving each image.
        '''

        self.__version__ = '0.1.0'

        self.df = df
        self.base_dir = base_dir.rstrip('/ ')

        self.transform = transform

    def __len__(self) -> int:
        '''
        Returns the number of data samples.

        Inputs: None.
        '''

        return self.df.shape[0]
    
    def __getitem__(self, index: int) -> Tuple:
        '''
        Retrieves the identity, image path, and image data at the passed index.

        Inputs:
            index: the index in the pandas DataFrame used in retrieving the image Tensors.

        Returns:
            identity: the string identity (ID) of the current image sample.
            path: the string filepath of the current image sample.
            img: the (transformed) image sample Tensor.
        '''

        identity, path = self.df.iloc[index]
        path = self.base_dir + '/' + path

        img = read_image(path=path)
        if self.transform:
            img = self.transform(img)

        assert img.shape[0] == 3, f'Image @ {path} is not RGB!'

        return (identity, path, img)
