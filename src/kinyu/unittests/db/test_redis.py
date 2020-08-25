from kinyu.db.api import kydb
from datetime import datetime
import os
import pytest


@pytest.fixture
def db():
    return kydb.connect('redis://{}:6379'.format(os.environ['REDIS_ENTRYPOINT']))


def test_redis(db):
    key = '/unittests/foo'
    db[key] = 123
    assert db[key] == 123
    assert db.read(key, reload=True) == 123


def test_redis_dict(db):
    key = '/unittests/dynamodb/bar'
    val = {
        'my_int': 123,
        'my_float': 123.456,
        'my_str': 'hello',
        'my_list': [1, 2, 3],
        'my_datetime': datetime.now()
    }
    db[key] = val
    assert db[key] == val
    assert db.read(key, reload=True) == val
