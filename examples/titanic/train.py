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

    df['Name'] = df['Name'] \
        .apply(lambda x: x.split(',')[-1].strip().split(' ')[0].replace('.', ''))
    df['Ticket'] = df['Ticket'] \
        .apply(adjust_ticket)
    df['Cabin'] = df['Cabin'] \
        .apply(adjust_cabin)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, 'model')
    data_config_dir = os.path.join(base_dir, 'data_config')
    input_file_name = os.path.join(base_dir, 'train.csv')
    hash_key = util.get_hash()

    # ------------------------------------------------------------------------------------------------------------------
    # Pre-process data
    # ------------------------------------------------------------------------------------------------------------------
    with open(os.path.join(base_dir, 'preprocess_config.json'), 'r') as f:
        data_config = json.load(f)

    data_handler = DataProcessor(train=True,
                                 config=data_config)
    data_handler.fetch_data(source_type='file',
                            input_type='csv',
                            file_path=input_file_name)
    data_handler.add_custom_func(func=custom_func)
    data_handler.data_processing()
    split_data, all_data = data_handler.get_training_data()
    util.save_data_config(handler=data_handler,
                          output_dir=data_config_dir,
                          suffix=hash_key)

    # ------------------------------------------------------------------------------------------------------------------
    # Model training
    # ------------------------------------------------------------------------------------------------------------------
    with open(os.path.join(base_dir, 'model_config.json'), 'r') as f:
        model_config = json.load(f)

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
    model_handler.save_model(output_dir=model_dir,
                             index=hash_key)


if __name__ == '__main__':
    main()
