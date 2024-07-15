import torch.nn as nn
import torch

class Accuracy(nn.Module):
    def __init__(self, dim=1):
        '''
        Initializes an instance of the Accuracy class.

        Inputs:
            dim: the dimension to perform the softmax and argmax over.
        '''

        super(Accuracy, self).__init__()

        self.__version__ = '0.1.1'

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        '''
        Calculates the accuracy, given the output class predictions of a classifier and the true labels.

        Inputs:
            y_pred: the output predictions of a classifier.
            y_true: the true labels.

        Returns:
            accuracy: the accuracy of the input classifier predictions.
        '''

        y_matches = (y_pred == y_true)
        accuracy = y_matches.sum().float() / float(y_true.shape[0])

        return accuracy
