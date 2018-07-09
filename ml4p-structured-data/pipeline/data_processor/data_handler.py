from __future__ import absolute_import
from .base_data_handler import BaseDataProcessor
from sklearn import preprocessing as sk_preprocess
from h2o.transforms import decomposition as h2o_decomposition
import h2o
import numpy as np
import pandas as pd
import pyodbc
import random


class DataProcessor(BaseDataProcessor):
    valid_source = {'file',
                    'db'}
    valid_input_type = {'csv',
                        'tsv'}
    valid_split = {'random'}
    valid_dim_reduction = {'pca'}
    valid_processing_method = {'DROP_COL',
                               'ONE_HOT_ENCODE_COL',
                               'ONE_N_ENCODE_COL',
                               'NORMALIZE_COL',
                               'STANDARDIZE_COL',
                               'IMPUTE_COL',
                               'DIM_REDUCTION',
                               'TRAIN_TEST_SPLIT'}

    def __init__(self):
        self.one_n_encoder = dict()
        self.normalizer = dict()
        self.standardizer = dict()
        self.imputer = dict()
        super(DataProcessor, self).__init__()

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
            raise ValueError("para must be a dictionary")
        if para:
            split_method = para.get('split_method')
            split_mode = para.get('split_mode')
            proportion = para.get('proportion')

            if split_method not in self.valid_split:
                raise ValueError("split_method {} is not implemented".format(split_method))
            if len(split_mode) != len(proportion):
                raise ValueError("split_mode and proportion must be of the same lengths")
            if abs(sum(proportion) - 1) > 1e-10:
                raise ValueError("Elements in proportion must sum to 1")

            if split_method == 'random':
                index_list = list(self.adj_data_df.index)
                record_count = len(index_list)
                random.shuffle(index_list)
                start_counter = 0
                last_counter = 0
                for name, p in zip(split_mode, proportion):
                    n = int(np.floor(record_count * p))
                    if last_counter != len(proportion) - 1:
                        temp_index_list = index_list[start_counter:(start_counter + n)]
                    else:
                        temp_index_list = index_list[start_counter:]
                    start_counter += n
                    self.split_data_dict[name] = self.adj_data_df.loc[temp_index_list]
                    last_counter += 1

    def data_processing(self, config):
        pipeline = config.get('PIPELINE')
        if pipeline is None:
            raise ValueError("PIPELINE is missing from the config")
        if type(pipeline) is not list:
            raise ValueError("PIPELINE must be a list")

        self.adj_data_df = self.raw_data_df.copy()
        for p in pipeline:
            method = list(p.keys())[0]
            para = p.get(method)
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
