from __future__ import absolute_import
from pipeline.data_processor import data_handler
from pipeline.supervised_learning import sl_model
import json
import os
import re


class TitanicDataProcessor(data_handler.DataProcessor):
    def __init__(self):
        super(TitanicDataProcessor, self).__init__()

    def _custom_processing_func(self, *args, **kwargs):
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

        self.adj_data_df['Name'] = self.adj_data_df['Name']\
            .apply(lambda x: x.split(',')[-1].strip().split(' ')[0].replace('.', ''))
        self.adj_data_df['Ticket'] = self.adj_data_df['Ticket']\
            .apply(adjust_ticket)
        self.adj_data_df['Cabin'] = self.adj_data_df['Cabin']\
            .apply(adjust_cabin)


if __name__ == '__main__':
    # base_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = "D:\\GitHub\\ml4p-structured-data\\examples\\titanic"
    input_file_name = os.path.join(base_dir, 'train.csv')

    # ------------------------------------------------------------------------------------------------------------------
    # Pre-process data
    # ------------------------------------------------------------------------------------------------------------------
    with open(os.path.join(base_dir, 'preprocess_config.json'), 'r') as f:
        data_config = json.load(f)

    data_handler = TitanicDataProcessor()
    data_handler.fetch_data(source_type='file',
                            input_type='csv',
                            file_path=input_file_name)
    data_pipeline = data_config.get('PIPELINE')
    data_handler.data_processing(pipeline=data_pipeline)
    split_data, all_data = data_handler.get_training_data()

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

    model_handler.init_model()
    model_handler.train_model()
    model_handler.save_model(key='XGB')
