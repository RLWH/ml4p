from __future__ import absolute_import
from abc import abstractmethod, ABCMeta
import pandas as pd


class BaseDataProcessor:
    __metaclass__ = ABCMeta

    def __init__(self):
        self.raw_data_df = pd.DataFrame()
        self.adj_data_df = pd.DataFrame()
        self.split_data = list()
        self.target_col = list()

    @abstractmethod
    def fetch_data(self, *args, **kwargs):
        pass

    @abstractmethod
    def data_processing(self, *args, **kwargs):
        pass

    def get_training_data(self):
        return self.split_data, self.target_col