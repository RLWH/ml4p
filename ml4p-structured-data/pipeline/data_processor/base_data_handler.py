from __future__ import absolute_import
from abc import abstractmethod, ABCMeta


class BaseDataProcessor:
    __metaclass__ = ABCMeta

    def __init__(self):
        self.raw_data_df = None
        self.adj_data_df = None
        self.split_data_dict = dict()

    @abstractmethod
    def fetch_data(self, *args, **kwargs):
        pass

    @abstractmethod
    def data_processing(self, *args, **kwargs):
        pass

    def get_split_data(self):
        return self.split_data_dict
