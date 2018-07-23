from __future__ import absolute_import
from pipeline.data_processor.data_handler import DataProcessor
from pipeline.supervised_learning import sl_model
import pipeline.misc.util as util
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


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_output_dir = os.path.join(base_dir, 'model_output')
    config_output_dir = os.path.join(base_dir, 'data_config_output')
    config_dir = os.path.join(base_dir, 'config')
    data_dir = os.path.join(base_dir, 'data')
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
    if not os.path.exists(config_output_dir):
        os.makedirs(config_output_dir)

    input_file_name = os.path.join(data_dir, 'train.csv')
    hash_key = util.get_hash()
    with open(os.path.join(config_dir, 'preprocess_config.json'), 'r') as f:
        data_config = json.load(f)
    with open(os.path.join(config_dir, 'model_config.json'), 'r') as f:
        model_config = json.load(f)

    # ------------------------------------------------------------------------------------------------------------------
    # Pre-process data
    # ------------------------------------------------------------------------------------------------------------------
    data_handler = DataProcessor(train=True,
                                 config=data_config)
    data_handler.fetch_data(source_type='file',
                            input_type='csv',
                            file_path=input_file_name)
    data_handler.add_custom_func(func=custom_func)
    data_handler.data_processing()
    split_data, all_data = data_handler.get_data()
    util.save_data_config(handler=data_handler,
                          output_dir=config_output_dir,
                          suffix=hash_key)

    # ------------------------------------------------------------------------------------------------------------------
    # Model training
    # ------------------------------------------------------------------------------------------------------------------
    model_pipeline = model_config.get('PIPELINE')
    model_handler = sl_model.TrainModelHandler()
    sl_model.BaseModel.set_data(split_data=split_data,
                                all_data=all_data)

    for m in model_pipeline:
        model_name = m.get('model')
        model_settings = m.get('settings')
        model_handler.add_model(model_name=model_name,
                                model_settings=model_settings)

    model_handler.train_model()
    model_handler.report_results()
    model_handler.save_model(output_dir=model_output_dir,
                             index=hash_key)


if __name__ == '__main__':
    main()
