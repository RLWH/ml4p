from __future__ import absolute_import
from pipeline.data_processor.data_handler import DataProcessor
import argparse
import pipeline.misc.util as util
import os


def main(sys_args):
    file_path = os.path.join(sys_args.base_dir,
                             sys_args.file_name)
    data_config = util.load_data_config(path=sys_args.base_dir,
                                        filename=sys_args.data_config_name)
    # model_config = util.load_data_config(path=sys_args.base_dir,
    #                                      filename=sys_args.model_config_name)

    # Process data from imported configs
    data_handler = DataProcessor(train=False)
    data_handler.load_from_dict(pickle_dict=data_config)
    data_handler.fetch_data(source_type='file',
                            input_type='csv',
                            file_path=file_path)
    data_handler.data_processing()
    data_handler.all_data['train_x'].to_csv('after.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument('--file_name', default='test.csv')
    parser.add_argument('--data_config_name', default='data_config_52cb905bd18745a7b6cf1b6ce331903e')
    parser.add_argument('--model_config_name')
    args = parser.parse_args()
    main(sys_args=args)
