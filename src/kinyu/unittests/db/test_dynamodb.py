from kinyu.db.api import kydb
from datetime import datetime
import os
import pytest

DYNAMODB = os.environ['KINYU_UNITTEST_DYNAMODB']


@pytest.fixture
def db():
    return kydb.connect('dynamodb://' + DYNAMODB)


def test_dynamodb(db):
    assert type(db).__name__ == 'DynamoDB'
    key = '/unittests/dynamodb/foo'
    db[key] = 123
    assert db[key] == 123
    assert db.read(key, reload=True) == 123


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
    assert db.read(key, reload=True) == val


def test_memory_with_basepath():
    db = kydb.connect(f'dynamodb://{DYNAMODB}/my/base/path')
    key = '/apple'
    db[key] = 123
    assert db[key] == 123
    assert db.read(key, reload=True) == 123
