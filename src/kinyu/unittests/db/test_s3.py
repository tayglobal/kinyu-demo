from kinyu.db.api import kydb
from datetime import datetime
import pytest


@pytest.fixture
def db():
    return kydb.connect('s3://epython')


def test_s3(db):
    key = '/unittests/s3/foo'
    db[key] = 123
    assert db[key] == 123


def test_s3_dict(db):
    key = '/unittests/s3/bar'
    val = {
        'my_int': 123,
        'my_float': 123.456,
        'my_str': 'hello',
        'my_list': [1, 2, 3],
        'my_datetime': datetime.now()
    }
    db[key] = val
    assert db[key] == val
