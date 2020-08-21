from kinyu.db.api import kydb
from datetime import datetime
import pytest


@pytest.fixture
def db():
    return kydb.connect('redis://ec2-52-212-162-219.eu-west-1.compute.amazonaws.com:6379')


def test_redis(db):
    key = '/unittests/foo'
    db[key] = 123
    assert db[key] == 123


def test_dynamodb_dict(db):
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
