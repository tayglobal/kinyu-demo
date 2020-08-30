from kinyu.db.api import kydb
from datetime import datetime
import pytest

BASE_URL = 'https://files.tayglobal.com/kinyu-demo'


@pytest.fixture
def db():
    """
    The fixture was prepared with this::

    with open('test_http_basic', 'w') as f:
        f.write(pickle.dumps(123).hex())

    val = {
        'my_int': 123,
        'my_float': 123.456,
        'my_str': 'hello',
        'my_list': [1, 2, 3],
        'my_datetime': datetime.now()
    }

    with open('test_http_dict', 'w') as f:
        f.write(pickle.dumps(val).hex())


    aws s3 cp test_http_basic s3://files.tayglobal.com/kinyu-demo/db/tests/
    aws s3 cp test_http_dict s3://files.tayglobal.com/kinyu-demo/db/tests/
    """
    return kydb.connect(BASE_URL)


def test_http_basic(db):
    key = '/db/tests/test_http_basic'
    assert db[key] == 123
    assert db.exists(key)


def test_http_not_exist(db):
    assert not db.exists('does_not_exist')


def test_http_dict(db):
    key = '/db/tests/test_http_dict'
    val = {
        'my_int': 123,
        'my_float': 123.456,
        'my_str': 'hello',
        'my_list': [1, 2, 3],
        'my_datetime': datetime(2020, 8, 30, 2, 5, 0, 580731)
    }
    assert db[key] == val
