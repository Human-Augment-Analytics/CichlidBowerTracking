from typing import Dict, List

class EpochLogger:
    def __init__(self, value_type='Loss'):
        '''
        Initializes an instance of the EpochLogger class.

        Inputs:
            value_type: a string indicating what types of values will be stored by the logger; should be input as you'd hope to see it on the y-axis of a graph (defaults to \'Loss\').
        '''

        self.__version__ = '0.1.0'

        self.value_type = value_type

        self.mins = []
        self.maxs = []
        self.avgs = []

    def add(self, minimum: float, maximum: float, average: float) -> None:
        '''
        Appends the passed min, max, and average values to the min, max, and average value logs, respectively.

        Inputs:
            minimum: a float representing the minimum value observed during a given epoch.
            maximum: a float representing the maximum value observed during a given epoch.
            average: a float representing the average value observed during a given epoch.

        Returns: None.
        '''
        
        self.mins.append(minimum)
        self.maxs.append(maximum)
        self.avgs.append(average)

    def get_logs(self) -> Dict[str, List[float]]:
        '''
        Returns a dictionary containing the (labelled) value logs being stored; meant for use in plotting values collected during training/validation.

        Inputs: None.

        Returns"
            logs: a dictionary containing the (labelled) value logs being stored.
        '''
        
        logs = {
            f'Minimum {self.value_type}': self.mins,
            f'Maximum {self.value_type}': self.maxs,
            f'Average {self.value_type}': self.avgs
        }

        return logs