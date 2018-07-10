from __future__ import absolute_import
from .base_model_handler import BaseModelHandler
import h2o
import xgboost as xgb


class XGBModel(BaseModelHandler):
    def init_model(self, *args, **kwargs):
        self.model = xgb.Booster(**kwargs)

    def load_model(self, *args, **kwargs):
        model_path = kwargs.get('model_path')
        if model_path is None:
            raise ValueError('model_path is missing')
        self.model.load_model(model_path)

    def train_model(self, *args, **kwargs):
        params = kwargs.get('params')
        num_boost_round = kwargs.get('num_boost_round')
        early_stopping_rounds = kwargs.get('early_stopping_rounds')
        if params is None:
            raise ValueError("params is missing")
        if num_boost_round is None:
            raise ValueError("num_boost_round is missing")
        if early_stopping_rounds is None:
            raise ValueError("early_stopping_rounds is missing")
        for s in self.split_data:
            train_data = s.get('train')
            train_x = train_data[-self.target_col]
            train_y = train_data[self.target_col]

            test_data = s.get('test')
            test_x = test_data[-self.target_col]
            test_y = test_data[self.target_col]

            d_train = xgb.DMatrix(train_x, label=train_y)
            d_test = xgb.DMatrix(test_x, label=test_y)

            watchlist = [(d_train, 'train'), (d_test, 'test')]
            progress = dict()

            model = xgb.train(params=params,
                              dtrain=d_train,
                              num_boost_round=100,
                              evals=watchlist,
                              early_stopping_rounds=50,
                              evals_result=progress)

            # TODO: Summarize validation metrics and return the best model to self.model

    def predict(self, *args, **kwargs):
        pass


class GBMModel(BaseModelHandler):
    def init_model(self, *args, **kwargs):
        pass

    def load_model(self, *args, **kwargs):
        pass

    def train_model(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass


class RFModel(BaseModelHandler):
    def init_model(self, *args, **kwargs):
        pass

    def load_model(self, *args, **kwargs):
        pass

    def train_model(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass


class AdaBoostModel(BaseModelHandler):
    def init_model(self, *args, **kwargs):
        pass

    def load_model(self, *args, **kwargs):
        pass

    def train_model(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass


class GLMModel(BaseModelHandler):
    def init_model(self, *args, **kwargs):
        pass

    def load_model(self, *args, **kwargs):
        pass

    def train_model(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass


class MLPModel(BaseModelHandler):
    def init_model(self, *args, **kwargs):
        pass

    def load_model(self, *args, **kwargs):
        pass

    def train_model(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass


class NaiveBayesModel(BaseModelHandler):
    def init_model(self, *args, **kwargs):
        pass

    def load_model(self, *args, **kwargs):
        pass

    def train_model(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass


class LDAModel(BaseModelHandler):
    def init_model(self, *args, **kwargs):
        pass

    def load_model(self, *args, **kwargs):
        pass

    def train_model(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass


class QDAModel(BaseModelHandler):
    def init_model(self, *args, **kwargs):
        pass

    def load_model(self, *args, **kwargs):
        pass

    def train_model(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass


class SVMModel(BaseModelHandler):
    def init_model(self, *args, **kwargs):
        pass

    def load_model(self, *args, **kwargs):
        pass

    def train_model(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass
