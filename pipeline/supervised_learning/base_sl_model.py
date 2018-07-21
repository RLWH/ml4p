from __future__ import absolute_import
from abc import abstractmethod, ABCMeta
from hyperopt.pyll.base import scope
import h2o
import hyperopt
import numpy as np
import pandas as pd


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


class BaseH2OModel(BaseModel):
    valid_metric = ['mse',
                    'rmse',
                    'mae',
                    'rmsle',
                    'r2',
                    'mean_residual_deviance',
                    'logloss',
                    'mean_per_class_error',
                    'null_degrees_of_freedom',
                    'residual_degrees_of_freedom',
                    'null_deviance',
                    'residual_deviance',
                    'aic',
                    'auc',
                    'gini'
                    ]
    categorical_distribution = ['bernoulli',
                                'multinomial']

    def __init__(self):
        super(BaseH2OModel, self).__init__()
        self.h2o_metric_fn = None
        self.h2o_estimator = None
        h2o.init()

    @staticmethod
    def get_metric(model, metric):
        if type(metric) is not str:
            raise ValueError("metric must be a string")
        lw_metric = metric.lower()
        if lw_metric not in BaseH2OModel.valid_metric:
            raise ValueError("{} is not a supported metric.".format(metric))
        else:
            eval_metric = getattr(model, metric)()
        return eval_metric

    def _eval_results(self, hp_params):
        loss_list = list()
        maximize = self.other_para['maximize']
        eval_metric_name = self.other_para['eval_metric_name']
        for s in BaseModel.split_data:
            train_x = s['train_x']
            train_y = s['train_y']
            test_x = s['test_x']
            test_y = s['test_y']
            d_train = h2o.H2OFrame(pd.concat([train_x, train_y], axis=1))
            d_test = h2o.H2OFrame(pd.concat([test_x, test_y], axis=1))
            X_name = list(train_x.columns)
            y_name = list(train_y.columns)[0]
            if hp_params.get('distribution') is not None:
                check = hp_params.get('distribution')
            elif hp_params.get('family') is not None:
                check = hp_params.get('family')
            else:
                check = None
            if check in BaseH2OModel.categorical_distribution or check is None:
                d_train[y_name] = d_train[y_name].asfactor()
                d_test[y_name] = d_test[y_name].asfactor()
            if hp_params.get('hidden') is not None:
                hp_params['hidden'] = list(hp_params['hidden'])
            model = self.h2o_estimator(**hp_params)
            model.train(X_name, y_name, training_frame=d_train, validation_frame=d_test)
            eval_metric = self.get_metric(model=model, metric=eval_metric_name)
            if maximize:
                loss = -1 * eval_metric
            else:
                loss = eval_metric
            loss_list.append(loss)
        output_dict = {'loss': np.average(loss_list),
                       'status': hyperopt.STATUS_OK}
        return output_dict

    def train_model(self, *args, **kwargs):
        params = kwargs.get('params')
        maximize = kwargs.get('maximize')
        eval_metric_name = kwargs.get('eval_metric')
        max_autotune_eval_rounds = kwargs.get('max_autotune_eval_rounds')

        if params is None:
            raise ValueError("params is missing")
        if maximize is None:
            raise ValueError("maximize is missing")
        if eval_metric_name is None:
            raise ValueError("eval_metric is missing")

        self.raw_model_para = params
        self.other_para['eval_metric_name'] = eval_metric_name
        self.other_para['maximize'] = maximize
        self.auto_tune_rounds = max_autotune_eval_rounds
        self.adj_params(after_ht=False)

        # Auto-tuning parameters
        if self.auto_tune:
            self.adj_params(after_ht=True)

        # Get evaluation metrics
        eval_dict = self._eval_results(hp_params=self.best_model_para)
        if maximize:
            metric_val = -1 * eval_dict['loss']
        else:
            metric_val = eval_dict['loss']
        self.eval = {'metric': self.other_para['eval_metric_name'],
                     'value': metric_val}

        # Train on all data
        all_train_x = BaseModel.all_data['train_x']
        all_train_y = BaseModel.all_data['train_y']
        X_name = list(all_train_x.columns)
        y_name = list(all_train_y.columns)[0]
        d_train_all = h2o.H2OFrame(pd.concat([all_train_x, all_train_y], axis=1))

        # Cast target column to factor if necessary
        if self.best_model_para.get('distribution') is not None:
            check = self.best_model_para.get('distribution')
        elif self.best_model_para.get('family') is not None:
            check = self.best_model_para.get('family')
        else:
            check = None
        if check in BaseH2OModel.categorical_distribution or check is None:
            d_train_all[y_name] = d_train_all[y_name].asfactor()
        self.model = self.h2o_estimator(**self.best_model_para)
        self.model.train(X_name, y_name, training_frame=d_train_all)

    def load_model(self, *args, **kwargs):
        pass

    def save_model(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass
