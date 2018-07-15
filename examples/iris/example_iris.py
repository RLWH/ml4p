from __future__ import absolute_import
from pipeline.data_processor import data_handler
from pipeline.supervised_learning import sl_model
import json
import os


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file_name = os.path.join(base_dir, 'iris.csv')

    # Pre-process data
    with open(os.path.join(base_dir, 'preprocess_config.json'), 'r') as f:
        data_config = json.load(f)

    data_handler = data_handler.DataProcessor()
    data_handler.fetch_data(source_type='file',
                            input_type='csv',
                            file_path=input_file_name)
    data_handler.data_processing(config=data_config)

    split_data, all_data = data_handler.get_training_data()

    # Train model
    with open(os.path.join(base_dir, 'model_config.json'), 'r') as f:
        model_config = json.load(f)

    target_col = model_config.get('TARGET_COL')
    exclude_col = model_config.get('EXCLUDE_COL')
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
