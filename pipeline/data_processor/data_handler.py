from __future__ import absolute_import
from sklearn import preprocessing as sk_preprocess
from h2o.transforms import decomposition as h2o_decomposition
import h2o
import numpy as np
import pandas as pd
import pyodbc
import random


class DataProcessor:
    valid_source = {'file',
                    'db'}
    valid_input_type = {'csv',
                        'tsv'}
    valid_split = {'random', 'n-fold'}
    valid_dim_reduction = {'pca'}
    valid_processing_method = {'DROP_COL',
                               'ONE_HOT_ENCODE_COL',
                               'ONE_N_ENCODE_COL',
                               'NORMALIZE_COL',
                               'STANDARDIZE_COL',
                               'IMPUTE_COL',
                               'DIM_REDUCTION',
                               'TRAIN_TEST_SPLIT',
                               'CUSTOM_FUNC'}

    def __init__(self):
        self.one_n_encoder = dict()
        self.normalizer = dict()
        self.standardizer = dict()
        self.imputer = dict()
        self.raw_data_df = pd.DataFrame()
        self.adj_data_df = pd.DataFrame()
        self.all_data = pd.DataFrame()
        self.split_data = list()

    def get_training_data(self):
        return self.split_data, self.all_data

    def _fetch_data_from_file(self, input_type, file_path, encoding, header):
        if input_type == 'csv':
            self.raw_data_df = pd.read_csv(file_path,
                                           encoding=encoding,
                                           header=header)
        elif input_type == 'tsv':
            self.raw_data_df = pd.read_csv(file_path,
                                           encoding=encoding,
                                           sep='\t',
                                           header=header)

    def _fetch_data_from_db(self, dsn, sql_query, encoding):
        con = pyodbc.connect(dsn=dsn)
        con.setencoding(encoding)
        cur = con.cursor()
        cur.execute(sql_query)
        col_name = [x[0] for x in cur.description]
        data = pd.DataFrame.from_records(cur.fetchall())
        if data.empty:
            data = pd.DataFrame(columns=col_name)
        else:
            data.columns = col_name
        self.raw_data_df = data

    def fetch_data(self, source_type, input_type, **kwargs):
        # File parameters
        file_path = kwargs.get('file_path', None)
        file_encoding = kwargs.get('file_encoding', 'utf-8')
        header = kwargs.get('header', 0)

        # Database parameters
        dsn = kwargs.get('dsn', None)
        sql_query = kwargs.get('sql_query', None)
        db_encoding = kwargs.get('db_encoding', 'utf-8')

        if source_type not in DataProcessor.valid_source:
            raise NotImplementedError("Source type is not supported")
        if input_type not in DataProcessor.valid_input_type:
            raise NotImplementedError("Input type is not supported")
        elif input_type in {'csv', 'tsv'} and file_path is None:
            raise ValueError('Specify file paths for csv / tsv data type')
        if source_type == 'db' and (dsn is None or sql_query is None):
            raise ValueError('dsn and sql_query must be non-empty if source_type is db')

        if source_type == 'file':
            self._fetch_data_from_file(input_type=input_type,
                                       file_path=file_path,
                                       encoding=file_encoding,
                                       header=header)
        elif source_type == 'db':
            self._fetch_data_from_db(dsn=dsn,
                                     sql_query=sql_query,
                                     encoding=db_encoding)

    def _drop_col(self, para):
        if type(para) is not list:
            raise ValueError("para must be a list")

        if para:
            self.adj_data_df.drop(para, axis=1, inplace=True)

    def _one_hot_encode(self, para):
        if type(para) is not list:
            raise ValueError("para must be a list")
        dummy_df_list = list()
        if para:
            for col in para:
                dummy = pd.get_dummies(self.adj_data_df[col], prefix=col)
                dummy_df_list.append(dummy)
            all_dummy_df = pd.concat(dummy_df_list, axis=1)
            self.adj_data_df.drop(para, axis=1, inplace=True)
            self.adj_data_df = pd.concat([self.adj_data_df, all_dummy_df], axis=1)

    def _one_n_encode(self, para):
        if type(para) is not list:
            raise ValueError("para must be a list")
        if para:
            for col in para:
                distinct_val_list = sorted(list(set(self.adj_data_df[col])))
                map_dict = {key: value for value, key in enumerate(distinct_val_list)}
                reverse_map_dict = {value: key for key, value in map_dict.items()}
                self.adj_data_df[col] = self.adj_data_df[col].apply(lambda x: map_dict.get(x))
                self.one_n_encoder[col] = reverse_map_dict

    def _normalize(self, para):
        if type(para) is not dict:
            raise ValueError("para must be a dictionary")
        if para:
            for col, settings in para.items():
                if settings == "default":
                    normalizer = sk_preprocess.Normalizer()
                else:
                    normalizer = sk_preprocess.Normalizer(**settings)
                normalized_values = normalizer.fit_transform(self.adj_data_df[col].values.reshape(1, -1))
                self.adj_data_df[col] = normalized_values[0]
                self.normalizer[col] = normalizer

    def _standardize(self, para):
        if type(para) is not dict:
            raise ValueError("para must be a dictionary")
        if para:
            for col, settings in para.items():
                if settings == "default":
                    standardizer = sk_preprocess.StandardScaler()
                else:
                    standardizer = sk_preprocess.StandardScaler(**settings)
                reshaped_list = self.adj_data_df[col].values.reshape(-1, 1)
                standardized_values = standardizer.fit_transform(reshaped_list)
                self.adj_data_df[col] = standardized_values.reshape(1, -1)[0]
                self.standardizer[col] = standardizer

    def _impute(self, para):
        if type(para) is not dict:
            raise ValueError("para must be a dictionary")
        if para:
            for col, settings in para.items():
                if settings == "default":
                    imputer = sk_preprocess.Imputer()
                else:
                    imputer = sk_preprocess.Imputer(**settings)
                reshaped_list = self.adj_data_df[col].values.reshape(-1, 1)
                imputed_values = imputer.fit_transform(reshaped_list)
                self.adj_data_df[col] = imputed_values.reshape(1, -1)[0]
                self.imputer[col] = imputer

    def _dim_reduction(self, para):
        if type(para) is not dict:
            raise ValueError("para must be a dictionary")
        if para:
            method = list(para.get('method').keys())[0]
            col_list = para.get('col_list')
            settings = para.get('method')[method]

            if len(para.get('method').keys()) > 1:
                raise ValueError("There can be only 1 dimensionality reduction method")
            if method not in self.valid_dim_reduction:
                raise ValueError("{} is not a valid dimensionality reduction method".format(method))
            if col_list != 'all' and type(col_list) is not list:
                raise ValueError("col_list is not valid")

            if col_list == 'all':
                col_list = list(self.adj_data_df.columns)

            if method == 'pca':
                h2o.init()
                h2o_frame = h2o.H2OFrame(self.adj_data_df[col_list])
                if settings == 'default':
                    h2o_pca = h2o_decomposition.H2OPCA()
                else:
                    h2o_pca = h2o_decomposition.H2OPCA(**settings)
                h2o_pca.train(x=col_list, training_frame=h2o_frame)
                pca_df = h2o_pca.predict(h2o_frame).as_data_frame()
                self.adj_data_df.drop(col_list, axis=1, inplace=True)
                self.adj_data_df = pd.concat([self.adj_data_df, pca_df], axis=1)
                h2o.cluster().shutdown()

    def _train_test_split(self, para):
        if type(para) is not dict:
            raise ValueError('para must be a dictionary')
        if para is None:
            raise ValueError('para cannot be None')

        split_method = para.get('split_method')
        split_proportion = para.get('split_proportion')
        target_col = para.get('target_col')
        exclude_col = para.get('exclude_col')

        if split_method is None:
            raise ValueError('split_method is missing')
        if split_proportion is None:
            raise ValueError('split_proportion is missing')
        if target_col is None:
            raise ValueError('target_col is missing')
        if split_proportion.get('train') is None or split_proportion.get('test') is None:
            raise ValueError('train or test proportion is missing')
        if split_method not in self.valid_split:
            raise ValueError("split_method {} is not implemented".format(split_method))
        if abs(split_proportion.get('train') + split_proportion.get('test') - 1) > 1e-10:
            raise ValueError("Elements in proportion must sum to 1")

        if exclude_col is None:
            feature_col = list(set(self.adj_data_df.columns) - set(target_col))
        else:
            feature_col = list(set(self.adj_data_df.columns) - set(target_col) - set(exclude_col))

        self.all_data = {'train_x': self.adj_data_df[feature_col],
                         'train_y': self.adj_data_df[target_col]}

        index_list = list(self.adj_data_df.index)
        record_count = len(index_list)
        train_prop = split_proportion.get('train')
        test_prop = split_proportion.get('test')
        n = int(np.floor(record_count * train_prop))

        if split_method == 'random':
            random.shuffle(index_list)
            train_index_list = index_list[0:n]
            test_index_list = index_list[n:]

            train_df = self.adj_data_df.loc[train_index_list]
            test_df = self.adj_data_df.loc[test_index_list]

            train_x = train_df[feature_col]
            test_x = test_df[feature_col]
            train_y = train_df[target_col]
            test_y = test_df[target_col]
            self.split_data.append({'train_x': train_x,
                                    'train_y': train_y,
                                    'test_x': test_x,
                                    'test_y': test_y})

        elif split_method == 'n-fold':
            num_fold = int(np.floor(1 / test_prop))
            batch_size = record_count // num_fold
            random.shuffle(index_list)
            for n in range(num_fold):
                if n != num_fold - 1:
                    test_index_list = index_list[n * batch_size: (n + 1) * batch_size]
                else:
                    test_index_list = index_list[n * batch_size:]
                train_index_list = list(set(index_list) - set(test_index_list))
                train_df = self.adj_data_df.loc[train_index_list]
                test_df = self.adj_data_df.loc[test_index_list]
                train_x = train_df[feature_col]
                test_x = test_df[feature_col]
                train_y = train_df[target_col]
                test_y = test_df[target_col]
                self.split_data.append({'train_x': train_x,
                                        'train_y': train_y,
                                        'test_x': test_x,
                                        'test_y': test_y})

    def _custom_processing_func(self, *args, **kwargs):
        """ To be overriden, if needed"""
        pass

    def data_processing(self, pipeline):
        self.adj_data_df = self.raw_data_df.copy()
        for p in pipeline:
            method = p.get('method')
            para = p.get('para')
            if method not in DataProcessor.valid_processing_method:
                raise ValueError("{} is not implemented".format(method))

            if method == 'DROP_COL':
                self._drop_col(para=para)
            elif method == 'ONE_HOT_ENCODE_COL':
                self._one_hot_encode(para=para)
            elif method == 'ONE_N_ENCODE_COL':
                self._one_n_encode(para=para)
            elif method == 'NORMALIZE_COL':
                self._normalize(para=para)
            elif method == 'STANDARDIZE_COL':
                self._standardize(para=para)
            elif method == 'IMPUTE_COL':
                self._impute(para=para)
            elif method == 'DIM_REDUCTION':
                self._dim_reduction(para=para)
            elif method == 'TRAIN_TEST_SPLIT':
                self._train_test_split(para=para)
            elif method == 'CUSTOM_FUNC':
                if para is not None:
                    self._custom_processing_func(**para)
                else:
                    self._custom_processing_func()
