from __future__ import absolute_import
from pipeline.data_processor.data_handler import DataProcessor
from pipeline.supervised_learning import sl_model
import argparse
import pipeline.misc.util as util
import os


def main(args):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, 'model_output')
    adj_model_dir = os.path.join(model_dir, args.hash_key)
    data_config_dir = os.path.join(base_dir, 'data_config_output')
    data_config_name = 'data_config_{}.dat'.format(args.hash_key)
    data_dir = os.path.join(base_dir, 'data')

    # Process data from imported configs
    data_config = util.load_data_config(input_dir=data_config_dir,
                                        filename=data_config_name)
    data_handler = DataProcessor(train=False)
    data_handler.load_from_dict(pickle_dict=data_config)
    data_handler.fetch_data(source_type='file',
                            input_type='csv',
                            file_path=os.path.join(data_dir, 'test.csv'))
    data_handler.data_processing()
    _, all_data = data_handler.get_data()

    # Make predictions from imported models
    model_handler = sl_model.TrainModelHandler()
    sl_model.BaseModel.set_data(split_data=None,
                                all_data=all_data)
    model_handler.load_model(input_dir=adj_model_dir)
    print(model_handler.model_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hash_key')
    args = parser.parse_args()
    main(args)
