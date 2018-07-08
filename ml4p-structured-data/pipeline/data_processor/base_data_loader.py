from __future__ import absolute_import
from abc import abstractmethod, ABCMeta


class BaseDataProcessor(metaclass=ABCMeta):
    @abstractmethod
    def fetch_data(self, *args, **kwargs):
        pass

    @abstractmethod
    def data_cleaning(self, *args, **kwargs):
        pass

    @abstractmethod
    def train_test_split(self, *args, **kwargs):
        pass
