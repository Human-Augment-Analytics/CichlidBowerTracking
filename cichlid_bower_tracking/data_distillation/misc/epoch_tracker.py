class EpochTracker:
    def __init__(self):
        '''
        Initializes an instance of the EpochTracker class.

        Inputs: None.
        '''

        self.__version__ = '0.1.0'

        self.max = float('-inf')
        self.min = float('inf')

        self.sum = 0.0
        self.nrecords = 0.0
        self.avg = 0.0

    def add(self, x_i: float) -> None:
        '''
        Adds the passed value to self.sum, increments self.nrecords, updates self.max and self.min, and computes self.avg.

        Inputs:
            x_i: the value to be added to the tracker.

        Returns: None.
        '''
        
        self.sum += x_i
        self.nrecords += 1

        self.max = max(x_i, self.max)
        self.min = min(x_i, self.min)

        self.avg = self.sum / self.nrecords