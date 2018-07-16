from __future__ import absolute_import
from .base_sl_model import BaseModel
from ..misc import util
from hyperopt.pyll.base import scope
import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
import hyperopt
import numpy as np
import xgboost as xgb
import os
import pandas as pd
import pickle


class TrainModelHandler:
    def __init__(self):
        self.model_dict = dict()
        self.setting_dict = dict()

    def add_model(self, model_name, model_settings):
        self.setting_dict[model_name] = model_settings
        if model_name == 'XGB':
            self.model_dict[model_name] = XGBModel()
        elif model_name == 'GBM':
            self.model_dict[model_name] = GBMModel()
        elif model_name == 'RF':
            self.model_dict[model_name] = RFModel()
        elif model_name == 'MLP':
            self.model_dict[model_name] = MLPModel()
        elif model_name == 'GLM':
            self.model_dict[model_name] = GLMModel()
        elif model_name == 'NB':
            self.model_dict[model_name] = NBModel()
        elif model_name == 'LDA':
            self.model_dict[model_name] = LDAModel()
        elif model_name == 'SVM':
            self.model_dict[model_name] = SVMModel()
        else:
            raise ValueError('Model {} is not implemented'.format(model_name))

    def init_model(self):
        for key in self.model_dict.keys():
            print('[{}] Initializing {}'.format(util.get_time_now(), key))
            settings = self.setting_dict[key]
            self.model_dict[key].init_model(**settings)

    def train_model(self):
        for key in self.model_dict.keys():
            print('[{}] Training {}'.format(util.get_time_now(), key))
            settings = self.setting_dict[key]
            self.model_dict[key].train_model(**settings)

    def eval_model(self):
        for key in self.model_dict.keys():
            print('[{}] Evaluating {}'.format(util.get_time_now(), key))
            self.model_dict[key].eval_model()

    def save_model(self, key):
        settings = self.setting_dict[key]
        self.model_dict[key].save_model(**settings)

    def del_model(self, model_name):
        del self.model_dict[model_name]


class XGBModel(BaseModel):
    def init_model(self, *args, **kwargs):
        self.model = xgb.Booster()

    def train_model(self, *args, **kwargs):
        params = kwargs.get('params')
        num_boost_round = kwargs.get('num_boost_round')
        early_stopping_rounds = kwargs.get('early_stopping_rounds')
        maximize = kwargs.get('maximize')
        verbose_eval = kwargs.get('verbose_eval')
        max_autotune_eval_rounds = kwargs.get('max_autotune_eval_rounds')

        if params is None:
            raise ValueError("params is missing")
        if num_boost_round is None:
            raise ValueError("num_boost_round is missing")
        if early_stopping_rounds is None:
            raise ValueError("early_stopping_rounds is missing")
        if maximize is None:
            raise ValueError("maximize is missing")

        # Adjust the format of params
        # TODO: Abstraction
        # TODO: Replace maximum
        adj_params = dict()
        int_params_list = list()
        auto_tune = False
        for k in params.keys():
            mode = params[k]['mode']
            values = params[k]['values']
            if mode == 'auto':
                auto_tune = True
                min_val = values.get('min')
                max_val = values.get('max')
                step = values.get('step')
                dtype = values.get('dtype')
                if dtype == 'int':
                    adj_params[k] = scope.int(hyperopt.hp.quniform(k, min_val, max_val, step))
                    int_params_list.append(k)
                elif dtype == 'float':
                    adj_params[k] = hyperopt.hp.uniform(k, min_val, max_val)
            elif mode == 'fixed':
                adj_params[k] = values
            else:
                raise ValueError('mode {} is not implemented'.format(mode))

        eval_metric_name = params['eval_metric']['values']

        def _eval_results(hp_params):
            loss_list = list()
            best_iter_list = list()
            for s in BaseModel.split_data:
                train_x = s['train_x']
                train_y = s['train_y']
                test_x = s['test_x']
                test_y = s['test_y']

                d_train = xgb.DMatrix(train_x, label=train_y)
                d_test = xgb.DMatrix(test_x, label=test_y)

                watchlist = [(d_train, 'train'), (d_test, 'test')]
                progress = dict()

                temp_model = xgb.train(params=hp_params,
                                       dtrain=d_train,
                                       num_boost_round=num_boost_round,
                                       evals=watchlist,
                                       early_stopping_rounds=early_stopping_rounds,
                                       evals_result=progress,
                                       maximize=maximize,
                                       verbose_eval=verbose_eval)

                best_iter = temp_model.best_iteration
                evals_test_result = progress.get('test')
                eval_metric = evals_test_result.get(eval_metric_name)[-1]
                print('{}: {} - Para: {}'.format(eval_metric_name, eval_metric, hp_params))
                if maximize:
                    loss = -1 * eval_metric
                else:
                    loss = eval_metric
                loss_list.append(loss)
                best_iter_list.append(best_iter)
            output_dict = {'loss': np.average(loss_list),
                           'best_iter': np.average(best_iter_list),
                           'status': hyperopt.STATUS_OK}
            return output_dict

        # Auto-tuning parameters
        # TODO: Abstraction
        if auto_tune:
            trials = hyperopt.Trials()
            best_params = hyperopt.fmin(fn=_eval_results,
                                        space=adj_params,
                                        algo=hyperopt.tpe.suggest,
                                        max_evals=max_autotune_eval_rounds,
                                        trials=trials)
            for key in best_params.keys():
                if key in int_params_list:
                    adj_params[key] = int(best_params[key])
                else:
                    adj_params[key] = float(best_params[key])

        # Get evaluation metrics
        eval_dict = _eval_results(hp_params=adj_params)
        best_iter = int(eval_dict['best_iter'])
        assert best_iter >= 0

        if maximize:
            metric_val = -1 * eval_dict['loss']
        else:
            metric_val = eval_dict['loss']
        self.eval = {'metric': eval_metric_name,
                     'value': metric_val}

        # Train on all data
        d_train_all = xgb.DMatrix(BaseModel.all_data['train_x'], label=BaseModel.all_data['train_y'])
        self.model = xgb.train(params=adj_params,
                               dtrain=d_train_all,
                               num_boost_round=best_iter + 1,
                               verbose_eval=True)

    def save_model(self, *args, **kwargs):
        model_path = kwargs.get('model_path')
        if model_path is None:
            raise ValueError('output_path cannot be empty')

        hash_code = util.get_hash()
        with open(os.path.join(model_path, 'xgb_{}.pickle'.format(hash_code)), 'wb') as f:
            pickle.dump(self.model, f, pickle.HIGHEST_PROTOCOL)

    def load_model(self, *args, **kwargs):
        model_path = kwargs.get('model_path')
        model_name = kwargs.get('model_name')
        if model_path is None:
            raise ValueError('model_path is missing')
        if model_name is None:
            raise ValueError('model_name is missing')
        with open(os.path.join(model_path, model_name), 'rb') as f:
            self.model = pickle.load(f)

    def predict(self, *args, **kwargs):
        data = kwargs.get('data')
        if data is None:
            raise ValueError('data cannot be empty')
        if self.model is None:
            raise ValueError('model cannot be empty. Train or load model first before making predictions')
        d_data = xgb.DMatrix(data)
        predictions = self.model.predict(d_data)
        return predictions


class GBMModel(BaseModel):
    def init_model(self, *args, **kwargs):
        self.model = H2OGradientBoostingEstimator()

    def train_model(self, *args, **kwargs):
        params = kwargs.get('params')

        if params is None:
            raise ValueError("params is missing")

        # Adjust the format of params
        adj_params = dict()
        int_params_list = list()
        auto_tune = False
        for k in params.keys():
            mode = params[k]['mode']
            values = params[k]['values']
            if mode == 'auto':
                auto_tune = True
                min_val = values.get('min')
                max_val = values.get('max')
                step = values.get('step')
                dtype = values.get('dtype')
                if dtype == 'int':
                    adj_params[k] = scope.int(hyperopt.hp.quniform(k, min_val, max_val, step))
                    int_params_list.append(k)
                elif dtype == 'float':
                    adj_params[k] = hyperopt.hp.uniform(k, min_val, max_val)
            elif mode == 'fixed':
                adj_params[k] = values
            else:
                raise ValueError('mode {} is not implemented'.format(mode))

        eval_metric_name = params['stopping_metric']['values']

        def _eval_results(hp_params):
            loss_list = list()
            best_iter_list = list()
            for s in BaseModel.split_data:
                train_x = s['train_x']
                train_y = s['train_y']
                test_x = s['test_x']
                test_y = s['test_y']

                d_train = h2o.H2OFrame(pd.concat([train_x, train_y], axis=1))
                d_test = h2o.H2OFrame(pd.concat([test_x, test_y], axis=1))

                gbm = H2OGradientBoostingEstimator(**hp_params)

                # Train the gbm model
                gbm.train(X_name, y_name, training_frame=d_train, validation_frame=d_test)
                # evals_test_result = progress.get('test')
                # eval_metric = evals_test_result.get(eval_metric_name)[-1]
                print('{}: {} - Para: {}'.format(eval_metric_name, eval_metric, hp_params))
                if maximize:
                    loss = -1 * eval_metric
                else:
                    loss = eval_metric
                loss_list.append(loss)
                best_iter_list.append(best_iter)
            output_dict = {'loss': np.average(loss_list),
                           'best_iter': np.average(best_iter_list),
                           'status': hyperopt.STATUS_OK}
            return output_dict

        # Auto-tuning parameters
        if auto_tune:
            trials = hyperopt.Trials()
            best_params = hyperopt.fmin(fn=_eval_results,
                                        space=adj_params,
                                        algo=hyperopt.tpe.suggest,
                                        max_evals=max_autotune_eval_rounds,
                                        trials=trials)
            for key in best_params.keys():
                if key in int_params_list:
                    adj_params[key] = int(best_params[key])
                else:
                    adj_params[key] = float(best_params[key])

        # Get evaluation metrics
        eval_dict = _eval_results(hp_params=adj_params)
        best_iter = int(eval_dict['best_iter'])
        assert best_iter >= 0

        if maximize:
            metric_val = -1 * eval_dict['loss']
        else:
            metric_val = eval_dict['loss']
        self.eval = {'metric': eval_metric_name,
                     'value': metric_val}

        # Train on all data
        d_train_all = xgb.DMatrix(BaseModel.all_data['train_x'], label=BaseModel.all_data['train_y'])
        self.model = xgb.train(params=adj_params,
                               dtrain=d_train_all,
                               num_boost_round=best_iter + 1,
                               verbose_eval=True)

    def save_model(self, *args, **kwargs):
        pass

    def load_model(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass


class RFModel(BaseModel):
    def init_model(self, *args, **kwargs):
        pass

    def load_model(self, *args, **kwargs):
        pass

    def train_model(self, *args, **kwargs):
        pass

    def save_model(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass


class GLMModel(BaseModel):
    def init_model(self, *args, **kwargs):
        pass

    def load_model(self, *args, **kwargs):
        pass

    def train_model(self, *args, **kwargs):
        pass

    def save_model(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass


class MLPModel(BaseModel):
    def init_model(self, *args, **kwargs):
        pass

    def load_model(self, *args, **kwargs):
        pass

    def train_model(self, *args, **kwargs):
        pass

    def save_model(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass


class NBModel(BaseModel):
    def init_model(self, *args, **kwargs):
        pass

    def load_model(self, *args, **kwargs):
        pass

    def train_model(self, *args, **kwargs):
        pass

    def save_model(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass


class LDAModel(BaseModel):
    def init_model(self, *args, **kwargs):
        pass

    def load_model(self, *args, **kwargs):
        pass

    def train_model(self, *args, **kwargs):
        pass

    def save_model(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass


class SVMModel(BaseModel):
    def init_model(self, *args, **kwargs):
        pass

    def load_model(self, *args, **kwargs):
        pass

    def train_model(self, *args, **kwargs):
        pass

    def save_model(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass
