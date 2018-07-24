import datetime
import uuid


def get_time_now(time_format='%Y%m%d %H:%M:%S'):
    return datetime.datetime.now().strftime(time_format)


def get_hash():
    return str(uuid.uuid4()).replace('-', '')
