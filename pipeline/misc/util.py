import datetime
import uuid
import marshal
import os
import pickle
import types


def get_time_now(time_format='%Y%m%d %H:%M:%S'):
    return datetime.datetime.now().strftime(time_format)


def get_hash():
    return str(uuid.uuid4()).replace('-', '')


def save_data_config(handler, output_dir='', suffix=None):
    dump_dict = handler.get_dict()
    dump_dict['custom_func'] = [marshal.dumps(x.__code__) for x in dump_dict['custom_func']]
    if suffix is None:
        file_name = 'data_config.dat'
    else:
        file_name = 'data_config_{}.dat'.format(suffix)
    with open(os.path.join(output_dir, file_name), 'wb') as f:
        pickle.dump(dump_dict, f, pickle.HIGHEST_PROTOCOL)


def load_data_config(filename, input_dir=''):
    with open(os.path.join(input_dir, filename), 'rb') as f:
        data_config = pickle.load(f)
    data_config['custom_func'] = [types.FunctionType(marshal.loads(x), globals()) for x in data_config['custom_func']]
    return data_config

