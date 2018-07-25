from __future__ import absolute_import
from pipeline.data_processor.data_handler import DataProcessor
from pipeline.supervised_learning import sl_model
import pipeline.misc.util as util
import argparse
import json
import os


def custom_func(df):
    import re

    def adjust_ticket(x):
        extraction = x.strip().split(' ')[0].replace('.', '').lower()
        if re.search('[a-z]', extraction):
            return extraction
        else:
            return 'others_{}'.format(len(x))

    def adjust_cabin(x):
        if type(x) is str:
            return x[0].lower()
        else:
            return 'others'

    df['Name'] = df['Name'].apply(lambda x: x.split(',')[-1].strip().split(' ')[0].replace('.', ''))
    df['Ticket'] = df['Ticket'].apply(adjust_ticket)
    df['Cabin'] = df['Cabin'].apply(adjust_cabin)


def main(sys_arg):
    data_config_path = sys_arg.data_config_path
    model_config_path = sys_arg.model_config_path
    hash_ = sys_arg.hash

    # ------------------------------------------------------------------------------------------------------------------
    # Pre-process data
    # ------------------------------------------------------------------------------------------------------------------
    data_handler = DataProcessor(train=True,
                                 config_path=data_config_path)
    data_handler.fetch_data()
    data_handler.add_custom_func(func=custom_func)
    data_handler.data_processing()
    split_data, all_data = data_handler.get_data()
    data_handler.save_data_config(hash_=hash_)

    # ------------------------------------------------------------------------------------------------------------------
    # Model training
    # ------------------------------------------------------------------------------------------------------------------
    model_handler = sl_model.ModelHandler(config=model_config)
    sl_model.BaseModel.set_data(split_data=split_data,
                                all_data=all_data)
    model_handler.add_model()
    model_handler.train_model()
    model_handler.report_results()
    model_handler.save_model(output_dir=model_output_dir,
                             index=hash_key)


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_config_path', default=os.path.join(base_dir, 'config', 'data_config.json'))
    parser.add_argument('--model_config_path', default=os.path.join(base_dir, 'config', 'model_config.json'))
    parser.add_argument('--hash', default=util.get_hash())
    args = parser.parse_args()
    main(sys_arg=args)
