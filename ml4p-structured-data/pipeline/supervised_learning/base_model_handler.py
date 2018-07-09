from __future__ import absolute_import
from abc import abstractmethod, ABCMeta


class BaseModelHandler:
    __metaclass__ = ABCMeta

    def __init__(self):
        self.model = None
        self.split_data_dict = None

    def load_data(self, split_data_dict):
        self.split_data_dict = split_data_dict

    @abstractmethod
    def init_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def train_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass
