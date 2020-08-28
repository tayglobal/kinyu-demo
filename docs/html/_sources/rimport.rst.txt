Remote Import
=============


Introduction
------------

Purpose of this module is to change Python’s import hooks from purely loading from local filesystem to loading some from remote system.
This makes it easy to rapidly test experimental code on compute clusters or quickly share and debug code form teammates.
The loading of remote source would be built on top of :ref:`kydb-page-ref`.


Example
-------

Create ``hello.py``::

    def greet():
         print("Hello World!")

Upload it to remote source

.. code-block:: bash

    python -m kinyu.rimport.uploader --srcdb=redis://my-redis.host/my-source --key=hello.py /tmp/hello.py
    
Now we can import the ``hello`` module::


    from kinyu.rimport.api import rimp
    rimp.set_srcdb('redis://my-redis.host/my-source')
    
    import hello
    
    hello.greet() # prints 'Hello Word!'
