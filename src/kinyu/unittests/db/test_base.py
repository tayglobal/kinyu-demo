from kinyu.db.base import BaseDB
import pytest


class DummyDb(BaseDB):
    def __init__(self, url: str):
        super().__init__(url)
        self.cache = {}
        self.raw_read_count = 0

    def get_raw(self, key):
        self.raw_read_count += 1
        return self.cache[key]

    def set_raw(self, key, value):
        self.cache[key] = value


@pytest.fixture
def db():
    return DummyDb('memory://unittests')


def test_cache(db):
    db = DummyDb('memory://unittests')
    key1 = '/unittests/cache/foo'
    key2 = '/unittests/cache/bar'
    db[key1] = 123
    db[key2] = 234

    assert db[key1] == 123
    assert db[key2] == 234

    assert db.raw_read_count == 0

    # Refresh only 1 key
    db.refresh(key1)
    assert db[key1] == 123
    assert db[key2] == 234
    assert db.raw_read_count == 1

    # Refresh all keys
    db.refresh()
    assert db[key1] == 123
    assert db[key2] == 234
    assert db.raw_read_count == 3


def test_exists(db):
    key = '/unittests/cache/foo'
    db[key] = 123
    assert db.exists(key)
    assert not db.exists('does_not_exist')


def test___get_name_and_basepath():
    assert BaseDB._get_name_and_basepath(
        's3://kinyu-demo') == ('kinyu-demo', '/')
    assert BaseDB._get_name_and_basepath(
        's3://kinyu-demo/') == ('kinyu-demo', '/')
    assert BaseDB._get_name_and_basepath(
        's3://kinyu-demo/foo') == ('kinyu-demo', '/foo/')
    assert BaseDB._get_name_and_basepath(
        's3://kinyu-demo/foo/') == ('kinyu-demo', '/foo/')
    assert BaseDB._get_name_and_basepath(
        's3://kinyu-demo/foo/bar') == ('kinyu-demo', '/foo/bar/')
    assert BaseDB._get_name_and_basepath(
        's3://kinyu-demo/foo/bar/') == ('kinyu-demo', '/foo/bar/')


def test__get_full_path():
    assert DummyDb('s3://kinyu-demo')._get_full_path('apple') == '/apple'
    assert DummyDb('s3://kinyu-demo/')._get_full_path('apple') == '/apple'

    assert DummyDb(
        's3://kinyu-demo/foo/bar')._get_full_path('apple') == '/foo/bar/apple'
    assert DummyDb(
        's3://kinyu-demo/foo/bar')._get_full_path('/apple') == '/foo/bar/apple'
    assert DummyDb(
        's3://kinyu-demo/foo/bar')._get_full_path('/apple/orange') == '/foo/bar/apple/orange'
    assert DummyDb(
        's3://kinyu-demo/foo/bar')._get_full_path('apple/orange') == '/foo/bar/apple/orange'
