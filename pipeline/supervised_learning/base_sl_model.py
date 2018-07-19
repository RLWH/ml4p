from __future__ import absolute_import
from abc import abstractmethod, ABCMeta
from hyperopt.pyll.base import scope
import hyperopt


class BaseModel:
    __metaclass__ = ABCMeta
    split_data = None
    all_data = None

    def __init__(self):
        self.model = None
        self.eval = None
        self.raw_model_para = dict()
        self.best_model_para = dict()
        self.other_para = dict()
        self.int_params_list = list()
        self.auto_tune = False
        self.auto_tune_rounds = None

    @classmethod
    def set_data(cls, split_data, all_data):
        cls.split_data = split_data
        cls.all_data = all_data

    def adj_params(self, after_ht):
        if type(after_ht) is not bool:
            raise ValueError('after_ht must be Boolean')

        if not after_ht:
            for k in self.raw_model_para.keys():
                mode = self.raw_model_para[k]['mode']
                values = self.raw_model_para[k]['values']
                if mode == 'auto':
                    self.auto_tune = True
                    min_val = values.get('min')
                    max_val = values.get('max')
                    step = values.get('step')
                    dtype = values.get('dtype')
                    if dtype == 'int':
                        self.best_model_para[k] = scope.int(hyperopt.hp.quniform(k, min_val, max_val, step))
                        self.int_params_list.append(k)
                    elif dtype == 'float':
                        self.best_model_para[k] = hyperopt.hp.uniform(k, min_val, max_val)
                elif mode == 'fixed':
                    self.best_model_para[k] = values
                else:
                    raise ValueError('mode {} is not implemented'.format(mode))
        else:
            trials = hyperopt.Trials()
            best_params = hyperopt.fmin(fn=self._eval_results,
                                        space=self.best_model_para,
                                        algo=hyperopt.tpe.suggest,
                                        max_evals=self.auto_tune_rounds,
                                        trials=trials)
            for key in best_params.keys():
                if key in self.int_params_list:
                    self.best_model_para[key] = int(best_params[key])
                else:
                    self.best_model_para[key] = float(best_params[key])

    @abstractmethod
    def _eval_results(self, hp_params):
        pass

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
