from kinyu.rimport.api import RemoteImporter
from subprocess import Popen, PIPE
import os
import pytest
import tempfile


SRCDB = 'files://tmp/unittests/test_uploader'


@pytest.fixture
def script_file():
    f = tempfile.NamedTemporaryFile('w', delete=False)
    f.close()

    yield f.name
    os.remove(f.name)


def test_uploader(script_file):
    key = 'test_uploader/hello.py'
    script = '''
def ff():
    print("I'm from a file")
'''

    with open(script_file, 'w') as f:
        f.write(script)

    command = ["python", "-m", "kinyu.rimport.uploader",
               "--srcdb=" + SRCDB,
               "--key=" + key,
               script_file]

    with Popen(command, stdout=PIPE) as proc:
        output = proc.stdout.read()

    assert proc.returncode == 0
    assert output.decode(
        'ascii') == f"Uploading {script_file} to {SRCDB}/{key}\n"

    ri = RemoteImporter(SRCDB)
    assert(ri.db[key]['code'] == script)
