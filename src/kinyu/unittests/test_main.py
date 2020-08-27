from kinyu.rimport.api import RemoteImporter
from subprocess import Popen, PIPE
import os

DYNAMODB = os.environ['KINYU_UNITTEST_DYNAMODB']


def test_main():
    srcdb = 'dynamodb://' + DYNAMODB
    ri = RemoteImporter(srcdb)
    ri.add_script("unittest/main_module.py", '''
def main():
    print("Hello World")
''')
    command = ["python", "-m", "kinyu.main",
               "--srcdb=" + srcdb, "unittest.main_module"]
    with Popen(command, stdout=PIPE) as proc:
        assert proc.stdout.read() == b'Hello World\n'
