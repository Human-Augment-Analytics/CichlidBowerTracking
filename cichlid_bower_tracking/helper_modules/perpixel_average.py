import torch

class PerPixelAverage():
    def __init__(self, channels: int, width: int, height: int, dtype=torch.int64):
        '''
        Initializes an instance of the PerPixelAverage class. Uses the inputs to define an empty PyTorch Tensor to contain the sum of pixel values,
        and a counter to represent the number of bboxes that are included in the sum.

        Inputs:
            channels: an int representing the number of channels to be considered; for an RGB image, should be 3.
            width: an int representing a fixed width used in constructing the average image.
            height: an int representing a fixed height used in constructing the average image.
            dtype: a PyTorch datatype used in defining the sum Tensor; defaults to torch.int64, only change to another PyTorch integer type when less precision is required.
        '''

        self.__version__ = '0.1.0'
        self.channels = channels
        self.width = width
        self.height = height

        self.sum = torch.empty((channels, width, height), dtype=dtype)
        self.counter = 0

    def add(self, scl_bbox: torch.Tensor) -> None:
        '''
        Adds a PyTorch Tensor of shape (self.channels, self.width, self.height) to the current sum of bbox images. Also increments the image counter.

        Inputs:
            scl_frame: a PyTorch Tensor representing the pixel values of a scaled (resized) bbox image.

        Returns: None.
        '''

        assert scl_bbox.shape == (self.channels, self.width, self.height) # assume fixed size for bbox images

        self.sum += scl_bbox
        self.counter += 1

    def avg(self) -> torch.Tensor:
        '''
        Computes the per-pixel average of the sum of bbox images by dividing the sum by the number of images added. Specifically, the averaged value of the pixel in channel
        c at coordinates (i, j) is given by p_{c, i, j} = (1 / counter) * SUM_{all bboxes}(p_{c, i, j}).

        Inputs: None
        
        Returns: A PyTorch Tensor of shape (self.channels, self.width, self.height) which contains the pixel values of the averaged bbox image.
        '''
        return (self.sum // self.counter)