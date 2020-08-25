from kinyu.db.api import kydb

def test_uniondb():
    db = kydb.connect('memory://db1;memory://db2')
    db1, db2 = db.dbs
    db1['/foo'] = 1
    db2['/foo'] = 2
    assert db['/foo'] == 1
    
    db['/foo'] = 3
    assert db['/foo'] == 3
    assert db1['/foo'] == 3
    assert db2['/foo'] == 2
    
    