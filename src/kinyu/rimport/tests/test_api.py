from kinyu.rimport.api import RemoteImporter, rimp
import os
import tempfile
import pytest


@pytest.fixture
def script_file():
    f = tempfile.NamedTemporaryFile('w', delete=False)
    f.close()
    yield f.name
    os.remove(f.name)


def test_install():
    ri = RemoteImporter('memory://cache001')
    ri.db['foo.py'] = {'code': '''
def unittest_add(x, y):
    return x + y
    '''}

    ri.install()

    from foo import unittest_add
    assert unittest_add(2, 3) == 5

    import foo
    assert foo.unittest_add(6, 7) == 13


def test_add_script():
    ri = RemoteImporter('memory://cache002')
    script = '''
def hello_world():
    print("Hello World)
'''
    ri.add_script('bar.py', script)
    assert ri.db['bar.py']['code'] == script


def test_add_script_from_file(script_file):
    script = '''
def ff():
    print("I'm from a file")
'''
    with open(script_file, 'w') as f:
        f.write(script)

    ri = RemoteImporter('memory://cache003')
    key = 'from_file.py'
    ri.add_script_from_file(key, script_file)
    assert ri.db[key]['code'] == script


def test_rimp():
    rimp.set_srcdb('memory://cache004')
    assert type(rimp.remote_importer.db).__name__ == 'MemoryDB'
