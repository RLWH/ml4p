from __future__ import absolute_import
from base_data_loader import BaseDataProcessor
import pandas as pd
import pyodbc


class DataProcessor(BaseDataProcessor):
    def __init__(self):
        self.raw_data_df = None
        self.split_data_dict = dict()
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
        header = kwargs.get('header', 'utf-8')

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
        self.raw_data_df.drop(col_list, axis=1, inplace=True)

    def _one_hot_encode(self, col_list):
        pass

    def _one_n_encode(self, col_list):
        pass

    def _normalize(self, col_list):
        pass

    def _standardize(self, col_list):
        pass

    def _impute(self, col_list):
        pass

    def data_cleaning(self, config):
        drop_col_list = config.get('DROP_COL')
        one_hot_encode_list = config.get('ONE_HOT_ENCODE_COL')
        one_n_encode_list = config.get('ONE_N_ENCODE_COL')
        normalize_list = config.get('NORMALIZE_COL')
        standardize_list = config.get('STANDARDIZE_COL')
        impute_list = config.get("IMPUTE_COL")

        self._drop_col(col_list=drop_col_list)
        self._one_hot_encode(col_list=one_hot_encode_list)
        self._one_n_encode(col_list=one_n_encode_list)
        self._normalize(col_list=normalize_list)
        self._standardize(col_list=standardize_list)
        self._impute(col_list=impute_list)

    def train_test_split(self):
        pass
