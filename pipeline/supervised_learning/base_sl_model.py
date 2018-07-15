from __future__ import absolute_import
from abc import abstractmethod, ABCMeta


class BaseModel:
    __metaclass__ = ABCMeta
    split_data = None
    all_data = None

    def __init__(self):
        self.model = None
        self.eval = None

    @classmethod
    def set_data(cls, split_data, all_data):
        cls.split_data = split_data
        cls.all_data = all_data

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
    def save_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass
