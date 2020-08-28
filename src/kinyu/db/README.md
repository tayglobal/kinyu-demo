# KYDB (Kinyu Database)

## Introduction

This is just a simple wrapper for various NoSQL Database. Currently it offers:

 * Simple factory. A single URL would define the database or union.
 * Caching
 * Union: i.e. multiple databases where:
   * Read would look for the object in order
   * Write always writes to the first (front) db
   
## Roadmap
 * Transactions:
   * Atomic writes and rollback
   * API into transaction log
 * Time travel
 * Indexing: Unify interface of various KYDB implmentations that already support indexing.
 
## Simple example with AWS S3
   
Connect to KYDB with AWS S3 as the implementation.

```python
db = kydb.connect('s3://my-kydb-bucket')
```

Writing to DB

```python
key = '/mytest/foo'
db[key] = 123
```

Reading from DB:

```python
db[key] # returns 123
```

The above actually performed no read from the S3 bucket because of the cache.
Let's force reload when we read.

```python
db.read(key, reload=True) # returns 123
```

A bit more complicated types:

```python
key = '/mytest/bar'
val = {
    'my_int': 123,
    'my_float': 123.456,
    'my_str': 'hello',
    'my_list': [1, 2, 3],
    'my_datetime': datetime.now()
}
db[key] = val

assert db.read(key, reload=True) == val
```

## DynamoDB

```python
db = kydb.connect('dynamodb://kydb')
```

## Redis

```python
db = kydb.connect('redis://cache.epythoncloud.io:6379')
```

Or simply:

```python
db = kydb.connect('redis://cache.epythoncloud.io')
```

## In-Memory

```python
db = kydb.connect('memory://cache001')
```

## Base Path

You can set the base path of a KYDB as follows:

```python
db = kydb.connect('dynamodb://my-source-db/home/tony.yum')
db['foo'] = 'Hello World!'
```

Now 'Hello World!' would be written to */home/tony.yum/foo* in the *my-source-db* DynamoDB.

Same applies to any other implmentations.

## Union

The URL used on *connect* can be a semi-colon separated string.

This would create a Union Database.

Connecting:

```python
db = kydb.connect('memory://unittest;s3://my-unittest-fixture')
```
OR

```python
db = kydb.connect('redis://hotfixes.epythoncloud.io:6379;dynamodb://my-prod-src-db')
```

Reading and writing:

```python
db1, db2 = db.dbs
db1['/foo'] = 1
db2['/bar'] = 2

(db['/foo'], db['/bar']) # return (1, 2)

# Although db2 has /foo, it is db1's /foo that the union returns
db2['/foo'] = 3
db['/foo'] # return 1

# writing always happens on the front db
db['/foo'] = 4
db1['/foo'] # returns 4
db2['/foo'] # returns 3
```
