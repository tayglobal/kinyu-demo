from kinyu.db.api import kydb
from datetime import datetime
import pytest


@pytest.fixture
def db():
    return kydb.connect('memory://unittest')


def test_memory(db):
    key = '/unittests/foo'
    db[key] = 123
    assert db[key] == 123
    db[key] = 456
    assert db[key] == 456
    assert db.read(key, reload=True) == 456
    db.delete(key)
    assert not db.exists(key)


def test_memory_errors(db):
    with pytest.raises(KeyError):
        db['does_not_exist']

    with pytest.raises(KeyError):
        db.delete('does_not_exist')


def test_memory_dict(db):
    key = '/unittests/bar'
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
    db = kydb.connect('memory://unittest/my/base/path')
    key = '/apple'
    db[key] = 123
    assert db[key] == 123
    assert db.read(key, reload=True) == 123
