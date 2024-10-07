class IntraEpochLogger:
    def __init__(self, value_type='General', num_logs=3):
        '''
        A class for tracking (possibly multiple) statistics within a single epoch.

        Inputs:
            value_type: the type of statistics to be stored by the class.
            num_logs: the number of statistics to keep logs for.
        '''

        self.__version__ = '0.1.0'

        self.value_type = value_type
        self.num_logs = num_logs

        self.logs = [[] for _ in range(self.num_logs)]

    def add(self, *values) -> None:
        '''
        Adds an arbitrary number of values to the arbitrary number of logs stored.

        Inputs:
            values: an arbitrary number of values; must have one value per statistic log.

        Returns: nothing.
        '''

        assert len(values) == self.num_logs, f'Invalid Input: Expected {self.num_logs} inputs (got {len(values)})'
        assert all(isinstance(value, float) for value in values), 'Inavlid Input: All input values must be of type "int" or "float".'

        for idx, value in enumerate(values):
            self.logs[idx].append(value)