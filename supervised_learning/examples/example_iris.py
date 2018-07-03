import os
import json
import pandas as pd
import sys

parent_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
util_path = os.path.join(parent_path, 'data_processor')
sys.path.append(parent_path)
sys.path.append(util_path)
from data_processor import data_loader

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file_name = os.path.join(base_dir, 'iris.csv')
    with open(os.path.join(base_dir, 'config.json'), 'r') as f:
        config_file = json.load(f)

    data_handler = data_loader.DataProcessor()
    data_handler.fetch_data(source_type='file',
                            input_type='csv',
                            file_path=input_file_name)
    data_handler.data_cleaning(config=config_file.get('DATA_LOAD'))
    print(data_handler.raw_data_df)
