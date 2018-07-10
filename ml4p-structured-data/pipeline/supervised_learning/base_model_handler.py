from __future__ import absolute_import
from abc import abstractmethod, ABCMeta


class BaseModelHandler:
    __metaclass__ = ABCMeta

    def __init__(self):
        self.model = None
        self.split_data = None
        self.target_col = None

    def load_data(self, split_data, target_col):
        self.split_data = split_data
        self.target_col = target_col

    @abstractmethod
    def init_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def load_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def train_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass
