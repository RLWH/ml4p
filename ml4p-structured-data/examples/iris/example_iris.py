from __future__ import absolute_import
from ...pipeline.data_processor import data_handler
import json
import os


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file_name = os.path.join(base_dir, 'iris.csv')
    with open(os.path.join(base_dir, 'config.json'), 'r') as f:
        config_file = json.load(f)

    data_handler = data_handler.DataProcessor()
    data_handler.fetch_data(source_type='file',
                            input_type='csv',
                            file_path=input_file_name)
    data_handler.data_processing(config=config_file)

    # Checking
    data_handler.adj_data_df.to_csv('check.csv', index=False)
