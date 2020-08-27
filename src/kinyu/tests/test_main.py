from kinyu.rimport.api import RemoteImporter
from subprocess import Popen, PIPE
import os
import pytest
import json

DYNAMODB = os.environ['KINYU_UNITTEST_DYNAMODB']
SRCDB = 'dynamodb://' + DYNAMODB


@pytest.fixture
def remote_importer():
    ri = RemoteImporter(SRCDB)
    return ri


def _run_test(args):
    command = ["python", "-m", "kinyu.main", "--srcdb=" + SRCDB] + args
    with Popen(command, stdout=PIPE) as proc:
        output = proc.stdout.read()
    assert proc.returncode == 0
    return output


def test_main(remote_importer):
    remote_importer.add_script("unittest/main_module.py", '''
def main():
    print("Hello World")
''')
    assert _run_test(["unittest.main_module"]) == b'Hello World\n'


def test_entry(remote_importer):
    remote_importer.add_script("unittest/test_entry.py", '''
def test_entry():
    print("Welcome")
''')
    assert _run_test(
        ["unittest.test_entry", "--entry=test_entry"]) == b'Welcome\n'


def test_args(remote_importer):
    remote_importer.add_script("unittest/test_args.py", '''
def main(a, b, c=3):
    assert a == 1
    assert b == 2
    assert c == 3
''')

    args = {'a': 1, 'b': 2}
    _run_test(["unittest.test_args", '--args=' + json.dumps(args)])
