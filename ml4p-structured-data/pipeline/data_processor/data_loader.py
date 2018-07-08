from __future__ import absolute_import
from .base_data_loader import BaseDataProcessor
import pandas as pd
import pyodbc
from sklearn import preprocessing as sk_preprocess


class DataProcessor(BaseDataProcessor):
    def __init__(self):
        self.raw_data_df = None
        self.adj_data_df = None
        self.split_data_dict = dict()
        self.one_n_encoder = dict()
        self.normalizer = dict()
        self.standardizer = dict()
        self.imputer = dict()
        self.valid_source = {'file', 'db'}
        self.valid_input_type = {'csv', 'tsv'}

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

        if source_type not in self.valid_source:
            raise NotImplementedError("Source type is not supported")
        if input_type not in self.valid_input_type:
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

    def _drop_col(self, col_list):
        self.adj_data_df.drop(col_list, axis=1, inplace=True)

    def _one_hot_encode(self, col_list):
        dummy_df_list = list()
        for col in col_list:
            dummy = pd.get_dummies(self.adj_data_df[col], prefix=col)
            dummy_df_list.append(dummy)
        all_dummy_df = pd.concat(dummy_df_list, axis=1)
        self.adj_data_df.drop(col_list, axis=1, inplace=True)
        self.adj_data_df = pd.concat([self.adj_data_df, all_dummy_df], axis=1)

    def _one_n_encode(self, col_list):
        for col in col_list:
            distinct_val_list = sorted(list(set(self.adj_data_df[col])))
            map_dict = {key: value for value, key in enumerate(distinct_val_list)}
            reverse_map_dict = {value: key for key, value in map_dict.items()}
            self.adj_data_df[col] = self.adj_data_df[col].apply(lambda x: map_dict.get(x))
            self.one_n_encoder[col] = reverse_map_dict

    def _normalize(self, col_list):
        for col, para in col_list.items():
            if para == "default":
                normalizer = sk_preprocess.Normalizer()
            else:
                normalizer = sk_preprocess.Normalizer(**para)
            normalized_values = normalizer.fit_transform(self.adj_data_df[col].values.reshape(1, -1))
            self.adj_data_df[col] = normalized_values[0]
            self.normalizer[col] = normalizer

    def _standardize(self, col_list):
        for col, para in col_list.items():
            if para == "default":
                standardizer = sk_preprocess.StandardScaler()
            else:
                standardizer = sk_preprocess.StandardScaler(**para)
            reshaped_list = self.adj_data_df[col].values.reshape(-1, 1)
            standardized_values = standardizer.fit_transform(reshaped_list)
            self.adj_data_df[col] = standardized_values.reshape(1, -1)[0]
            self.standardizer[col] = standardizer

    def _impute(self, col_list):
        for col, para in col_list.items():
            if para == "default":
                imputer = sk_preprocess.Imputer()
            else:
                imputer = sk_preprocess.Imputer(**para)
            reshaped_list = self.adj_data_df[col].values.reshape(-1, 1)
            imputed_values = imputer.fit_transform(reshaped_list)
            self.adj_data_df[col] = imputed_values.reshape(1, -1)[0]
            self.imputer[col] = imputer

    def data_cleaning(self, config):
        drop_col_list = config.get('DROP_COL')
        one_hot_encode_list = config.get('ONE_HOT_ENCODE_COL')
        one_n_encode_list = config.get('ONE_N_ENCODE_COL')
        normalize_list = config.get('NORMALIZE_COL')
        standardize_list = config.get('STANDARDIZE_COL')
        impute_list = config.get("IMPUTE_COL")

        self.adj_data_df = self.raw_data_df.copy()
        if drop_col_list:
            self._drop_col(col_list=drop_col_list)
        if one_hot_encode_list:
            self._one_hot_encode(col_list=one_hot_encode_list)
        if one_n_encode_list:
            self._one_n_encode(col_list=one_n_encode_list)
        if normalize_list:
            self._normalize(col_list=normalize_list)
        if standardize_list:
            self._standardize(col_list=standardize_list)
        if impute_list:
            self._impute(col_list=impute_list)

    def train_test_split(self):
        pass