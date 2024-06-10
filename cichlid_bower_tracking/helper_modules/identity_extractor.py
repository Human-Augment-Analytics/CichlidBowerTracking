from typing import Dict, List
import os

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from torchvision.io import read_image
import torchvision.models
import torch

import numpy as np

class IdentityExtractor:
    def __init__(self, bboxes_dir: str, model_str: str, n_components: int, n_clusters: int, whiten=True, max_iter=300, random_state=42, dtype=torch.uint64):
        '''
        Initializes an instance of the IdentityExtractor class.

        Inputs:
            bboxes_dir: a path to the directory in which the transformed bbox images are stored, as performed by a BBoxCollector.
            model_str: a string indicating the type of pretrained model to be used in extracting features; must be one of ['vgg11', 'vgg11_bn', 
                       'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn'].
            n_components: an integer indicating the number of components into which the model output will be decomposed into using PCA.
            n_clusters: an integer indicating the expected number of clusters; should be equal to the number of fish in the tank.
            whiten: a Boolean indicating whether to whiten the data before passing through PCA; defaults to True.
            max_iter: an integer indicating the maximum number of iterations allowed for K-Means clustering; defualts to 300.
            random_state: an integer used in seeding Scikit-learn class instances; defaults to 42.
            dtype: a PyTorch dtype which indicates the precision which should be used in storing the bbox images as Tensors; defaults to uint64.
        '''

        self.__version__ = '0.1.0'

        # define dataset variables
        self.bboxes_dir = bboxes_dir
        self.bbox_files = self._collect_files()
        self.data = self._create_dataset(self.bbox_files)

        # define tracking variables
        self.activation = dict()
        self.features = dict()
        self.decompositions = dict()
        self.identities = dict()
        
        # define model based on model_str
        self.model_str = model_str
        if model_str == 'vgg11':
            self.model = torchvision.models.vgg11(pretrained=True)
        elif model_str == 'vgg11_bn':
            self.model = torchvision.models.vgg11_bn(pretrained=True)
        elif model_str == 'vgg13':
            self.model = torchvision.models.vgg13(pretrained=True)
        elif model_str == 'vgg13_bn':
            self.model = torchvision.models.vgg13_bn(pretrained=True)
        elif model_str == 'vgg16':
            self.model = torchvision.models.vgg16(pretrained=True)
        elif model_str == 'vgg16_bn':
            self.model = torchvision.models.vgg16_bn(pretrained=True)
        elif model_str == 'vgg19':
            self.model = torchvision.models.vgg19(pretrained=True)
        elif model_str == 'vgg19_bn':
            self.model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            self.model = torchvision.models.alexnet(pretrained=True)

        # define post-feature extraction pipeline
        self.pca = PCA(n_components=n_components, whiten=whiten, random_state=random_state)
        self.kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=random_state)

    def _collect_files(self) -> List[str]:
        bbox_files = []

        os.chdir(self.bboxes_dir)
        with os.scandir(self.bboxes_dir) as files:
            for file in files:
                assert file.name.split('.')[-1] in ['png', 'jpg', 'jpeg']

                bbox_files.append(file)

        return bbox_files
    
    def _create_dataset(self, bbox_files: List[str]) -> Dict:
        data = {}
        
        for file in bbox_files:
            bbox_img = read_image(file).numpy()
            data[file] = bbox_img

        return data

    def _get_activation(self, name: str):
        '''
        Sets up a PyTorch hook for getting the second-to-last (non-dropout) activation from the model.

        Inputs:
            name: a string indicating the key name in the activation dictionary.
        '''

        def hook(model, input, output):
            '''
            A PyTorch hook for getting the second-to-last (non-dropout) activation from the model.
            '''

            self.activation[name] = output.detach()

        return hook
    
    def _get_single_img_features(self, img: torch.Tensor) -> torch.Tensor:
        '''
        Calls the previously-defined 
        '''
        if 'vgg' in self.model_str:
            self.model.classifier[4].register_forward_hook(self._get_activation('fc2'))
        else:
            self.model.classifier[5].register_forward_hook(self._get_activation('fc2'))

        _ = self.model(img)

        return self.activation['fc2']
    
    def _get_all_features(self) -> None:
        for track_id, bbox in self.bboxes.items():
            featureset = self._get_single_img_features(torch.tensor(bbox, dtype=torch.uint64))

            self.features[track_id] = featureset.numpy()


