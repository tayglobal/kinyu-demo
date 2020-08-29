Developer
=========

This page is for developers of Kinyu Demo.

Style Guide
-----------

`PEP8 <https://www.python.org/dev/peps/pep-0008/>`_ Please.

Use your favouite IDE plugin or at least use `autopep8 <https://github.com/hhatto/autopep8>`_.

If you want to be fancy add Git pre-commit hooks to ensure formatting.


Unittest
--------

Please ensure all unittests passes before PR.

Note that :ref:`kydb-page-ref` unittests does require certain dbs to be setup.
You would need to set the following environment variables:

 * ``KINYU_UNITTEST_S3_BUCKET``: name of the S3 Bucket
 
 * ``KINYU_UNITTEST_DYNAMODB``: name of the DynamoDB
 
 * ``KINYU_UNITTEST_REDIS_HOST``: host of the redis server. Can also end with :port if non standard port. i.e. redis-host:8765

Docker
------

You will find the ``Dockerfile`` in the project root directory.


Building the container
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    docker build --tag kinyu-demo:0.1 .

List the images and we should see kinyu-demo

.. code-block:: bash

    docker images

Bash the container
^^^^^^^^^^^^^^^^^^

Find the docker

.. code-block:: bash

    docker images

Execute bash on it:

.. code-block:: bash

    docker exec -it <container name> /bin/bash

Docker Hub
^^^^^^^^^^

Login to Docker Hub

.. code-block:: bash

    docker login --username=yourhubusername --email=youremail@company.com

Tag it:

.. code-block:: bash

    docker tag bb38976d03cf yourhubusername/kinyu-demo:0.1


Push it:

.. code-block:: bash

    docker push yourhubusername/kinyu-demo

You can now delete all the local docker images and pull it:

.. code-block:: bash

    docker pull tayglobal/kinyu-demo:0.1

ECR
---

Authenticate:

.. code-block:: bash

    aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <aws_account_id>.dkr.ecr.<region>.amazonaws.com

Create repository:

.. code-block:: bash

    aws ecr create-repository \
        --repository-name kinyu-demo \
        --image-scanning-configuration scanOnPush=true \
        --region eu-west-1

    docker tag tayglobal/kinyu-demo:0.1 499030764380.dkr.ecr.eu-west-1.amazonaws.com/kinyu-demo:0.1
    
    docker push 499030764380.dkr.ecr.eu-west-1.amazonaws.com/kinyu-demo:0.1