from __future__ import absolute_import
from .base_sl_model import BaseModel, BaseH2OModel, BaseSKModel
from ..misc import util
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.naive_bayes import H2ONaiveBayesEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC, SVR
import glob
import hyperopt
import json
import numpy as np
import os
import xgboost as xgb


class ModelHandler:
    valid_model_list = ['XGB',
                        'GBM',
                        'RF',
                        'MLP',
                        'GLM',
                        'NB',
                        'LDA',
                        'QDA',
                        'SVC',
                        'SVR']
    valid_task = ['classification',
                  'regression']

    def __init__(self, config_path):
        self.model_dict = dict()
        self.setting_dict = dict()
        self.pred_dict = dict()
        self.config_path = config_path
        if not os.path.isfile(path=config_path):
            raise FileNotFoundError("Data config not found")
        else:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            if type(self.config) is not dict:
                raise ValueError("config must be a dictionary")
            self.model_pipeline = self.config.get('PIPELINE')
            self.task = self.config.get('TASK')
            if self.model_pipeline is None:
                raise ValueError('PIPELINE is not found in config')
            if self.task is None:
                raise ValueError('TASK is not found in config')

    def add_model(self):
        for model_name, model_settings in self.model_pipeline.items():
            if model_name not in ModelHandler.valid_model_list:
                print('Model {} is not implemented'.format(model_name))
            else:
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
                elif model_name == 'QDA':
                    self.model_dict[model_name] = QDAModel()
                elif model_name == 'SVC':
                    self.model_dict[model_name] = SVCModel()
                elif model_name == 'SVR':
                    self.model_dict[model_name] = SVRModel()

    def train_model(self):
        for key in self.model_dict.keys():
            print('[{}] Training {}'.format(util.get_time_now(), key))
            settings = self.setting_dict[key]
            self.model_dict[key].train_model(**settings)

    def report_results(self):
        for key in self.model_dict.keys():
            print('{}: {}'.format(key, self.model_dict[key].eval))

    def save_model(self, output_dir, index=util.get_hash(), save_all=True, key=None):
        if not save_all and (key is None or key not in self.model_dict.keys()):
            raise ValueError("Invalid model key")
        adj_output_dir = os.path.join(output_dir, index)
        if not os.path.exists(adj_output_dir):
            os.makedirs(adj_output_dir)
        if save_all:
            for k in self.model_dict.keys():
                self.model_dict[k].save_model(output_dir=adj_output_dir,
                                              filename=k)
        else:
            self.model_dict[key].save_model(output_dir=adj_output_dir,
                                            filename=key)

    def load_model(self, input_dir):
        model_path_list = glob.glob(os.path.join(input_dir, '*'))
        for p in model_path_list:
            model_key = os.path.basename(p)
            dir_name = os.path.dirname(p)
            self.add_model(model_name=model_key)
            self.model_dict[model_key].load_model(input_dir=dir_name,
                                                  filename=model_key)

    def predict(self):
        for key in self.model_dict.keys():
            self.pred_dict[key] = self.model_dict[key].predict()

    def del_model(self, model_name):
        del self.model_dict[model_name]


class XGBModel(BaseModel):
    def _eval_results(self, hp_params):
        loss_list = list()
        best_iter_list = list()
        num_boost_round = self.other_para['num_boost_round']
        early_stopping_rounds = self.other_para['early_stopping_rounds']
        maximize = self.other_para['maximize']
        verbose_eval = self.other_para['verbose_eval']
        eval_metric_name = self.other_para['eval_metric_name']
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
            best_iter = int(temp_model.best_iteration)
            evals_test_result = progress.get('test')
            eval_metric = evals_test_result.get(eval_metric_name)[best_iter]
            if maximize:
                loss = -1 * eval_metric
            else:
                loss = eval_metric
            loss_list.append(loss)
            best_iter_list.append(best_iter)
        output_dict = {'loss': np.average(loss_list),
                       'best_iter': int(np.average(best_iter_list)),
                       'status': hyperopt.STATUS_OK}
        return output_dict

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

        self.raw_model_para = params
        self.other_para['eval_metric_name'] = params['eval_metric']['values']
        self.other_para['num_boost_round'] = num_boost_round
        self.other_para['early_stopping_rounds'] = early_stopping_rounds
        self.other_para['maximize'] = maximize
        self.other_para['verbose_eval'] = verbose_eval
        self.auto_tune_rounds = max_autotune_eval_rounds
        self.adj_params(after_ht=False)

        # Auto-tuning parameters
        if self.auto_tune:
            self.adj_params(after_ht=True)

        # Get evaluation metrics
        eval_dict = self._eval_results(hp_params=self.best_model_para)
        best_iter = int(eval_dict['best_iter'])
        assert best_iter >= 0
        if maximize:
            metric_val = -1 * eval_dict['loss']
        else:
            metric_val = eval_dict['loss']
        self.eval = {'metric': self.other_para['eval_metric_name'],
                     'value': metric_val}

        # Train on all data
        BaseModel.all_data['train_x'].to_csv('check_1.csv')
        d_train_all = xgb.DMatrix(BaseModel.all_data['train_x'], label=BaseModel.all_data['train_y'])
        self.model = xgb.train(params=self.best_model_para,
                               dtrain=d_train_all,
                               num_boost_round=best_iter + 1,
                               verbose_eval=True)

    def predict(self, data=None):
        if data is None:
            data = BaseModel.all_data.get('train_x')
        if self.model is None:
            raise ValueError('model cannot be empty. Train or load model first before making predictions')
        d_data = xgb.DMatrix(data)
        pred = self.model.predict(d_data)
        return pred


class GBMModel(BaseH2OModel):
    def __init__(self):
        super(GBMModel, self).__init__()
        self.h2o_estimator = H2OGradientBoostingEstimator


class RFModel(BaseH2OModel):
    def __init__(self):
        super(RFModel, self).__init__()
        self.h2o_estimator = H2ORandomForestEstimator


class MLPModel(BaseH2OModel):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.h2o_estimator = H2ODeepLearningEstimator


class GLMModel(BaseH2OModel):
    def __init__(self):
        super(GLMModel, self).__init__()
        self.h2o_estimator = H2OGeneralizedLinearEstimator


class NBModel(BaseH2OModel):
    def __init__(self):
        super(NBModel, self).__init__()
        self.h2o_estimator = H2ONaiveBayesEstimator


class LDAModel(BaseSKModel):
    def __init__(self):
        super(LDAModel, self).__init__()
        self.sk_estimator = LinearDiscriminantAnalysis


class QDAModel(BaseSKModel):
    def __init__(self):
        super(QDAModel, self).__init__()
        self.sk_estimator = QuadraticDiscriminantAnalysis


class SVCModel(BaseSKModel):
    def __init__(self):
        super(SVCModel, self).__init__()
        self.sk_estimator = SVC


class SVRModel(BaseSKModel):
    def __init__(self):
        super(SVRModel, self).__init__()
        self.sk_estimator = SVR
