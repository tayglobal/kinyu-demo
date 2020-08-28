Main
====


Introduction
------------

This is the entry point to the Kinyu Demo system.
There need only one container registered on Docker Hub or ECR and we can run any code on a cluster.
There are no need to rebuild the container on code change, making experimental code much easier to test.

Local Conatiner
---------------

Get the Docker container

.. code-block:: bash

    docker pull tayglobal/kinyu-demo:0.1

Create ``hello.py``::

    def main():
         print("Hello World!")

Upload it to remote source

.. code-block:: bash

    python -m kinyu.rimport.uploader --srcdb=redis://my-redis-host/my-source --key=hello.py /tmp/hello.py


Run container pointing at that source:

.. code-block:: bash

    docker run tayglobal/kinyu-demo:0.1 python -m kinyu.main --srcdb=redis://my-redis-host hello

And we would see printing of  *Hello World!*. You can update the code on remote source db and repeat to see different results.


Run on ECS Cluster
------------------

With *hello.py* still in the remote source db, we can launch a task on ECS Cluster::

    client = boto3.client('ecs')
    
    response = client.run_task(
        launchType='FARGATE',
        cluster='kinyu-demo',
        taskDefinition='kinyu-demo',
        group='family:kinyu-demo',
        networkConfiguration={
            'awsvpcConfiguration': {
                'subnets': ['subnet-073c0a61'],
                'securityGroups': ['sg-05e5e111abfff2f77'],
                'assignPublicIp': 'ENABLED'
            }
        },
        
        overrides={
            'containerOverrides': [
                {
                    'name': 'kinyu-demo',
                    'command': ['python', '-m', 'kinyu.main',
                        '--srcdb=my-redis-host',
                        'hello']
                }
            ]
        }
    )


