import datetime
import uuid
import pickle


def get_time_now(time_format='%Y%m%d %H:%M:%S'):
    return datetime.datetime.now().strftime(time_format)


def get_hash():
    return str(uuid.uuid4()).replace('-', '')


def save_agent(agent, path):
    with open(path, 'wb') as f:
        pickle.dump(agent, f, pickle.HIGHEST_PROTOCOL)


def load_agent(path):
    with open(path, 'rb') as f:
        agent = pickle.load(f)
    return agent
