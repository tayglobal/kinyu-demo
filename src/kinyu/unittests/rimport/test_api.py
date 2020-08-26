from kinyu.db.api import kydb
from kinyu.rimport.api import RemoteImporter


def test_remote_importer():
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
