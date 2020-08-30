from kinyu.db.api import kydb
from datetime import datetime
import pytest
import tempfile

LOCAL_DIR = tempfile.gettempdir() + '/unittests/test_fs'


@pytest.fixture
def db():
    return kydb.connect('files:/' + LOCAL_DIR)


def test__get_fs_path():
    # db = kydb.connect('files:/' + LOCAL_DIR)
    # assert db._get_fs_path('foo') == LOCAL_DIR + '/foo'
    # assert db._get_fs_path('/foo') == LOCAL_DIR + '/foo'

    db = kydb.connect('files:/' + LOCAL_DIR + '/base/path')
    assert db._get_fs_path('foo') == LOCAL_DIR + '/base/path/foo'
    assert db._get_fs_path('/foo') == LOCAL_DIR + '/base/path/foo'


def test_files_basic(db):
    key = '/unittests/files/foo'
    db[key] = 123
    assert db[key] == 123
    assert db.exists(key)
    db.delete(key)
    assert not db.exists(key)


def test_files_dict(db):
    key = '/unittests/files/bar'
    val = {
        'my_int': 123,
        'my_float': 123.456,
        'my_str': 'hello',
        'my_list': [1, 2, 3],
        'my_datetime': datetime.now()
    }
    db[key] = val
    assert db[key] == val
