import click
import sys
import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent
from kinyu.db.api import kydb
from .api import rimp
import logging
import logging.handlers


logger = logging.getLogger(__name__)


class CodeSync(FileSystemEventHandler):
    def __init__(self, srcdb: str, localpath: str, recursive=True):
        self.db = kydb.connect(srcdb)
        self.localpath = localpath
        self.recursive = recursive

    def watch(self):
        observer = Observer()
        observer.schedule(self, self.localpath, recursive=self.recursive)
        observer.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()

    def on_modified(self, event):
        if isinstance(event, FileModifiedEvent):
            path = self._get_py_script(event.src_path)

            # Not a python script
            if not path:
                return

            with open(event.src_path, 'r') as f:
                code = f.read()

            # Started to write the file, but not finished. ignore this event.
            if not len(code):
                return

            logger.info('on_modified')
            logger.info(event.src_path)
            logger.info(code)

    def on_created(self, event):
        path = self._get_py_script(event.src_path)
        if path:
            logger.info('Created!')
            logger.info(path)

    def on_deleted(self, event):
        path = self._get_py_script(event.src_path)
        if path:
            logger.info('Deleted!')
            logger.info(path)

    def on_moved(self, event):
        # <FileMovedEvent: src_path='./hello2.py', dest_path='./hello.py'>
        src_script = self._get_py_script(event.src_path)
        if src_script:
            logger.info('Moved!')
            logger.info(f'Deleting {src_script} form remote db')

        dest_script = self._get_py_script(event.dest_path)
        if dest_script:
            logger.info('Moved!')
            logger.info(f'Writing {dest_script} to remote db')

    def _get_filename(self, path: str):
        return path.rsplit('/', 1)[1]

    def _get_py_script(self, path: str):
        """
        Returns filepath relative to localpath if python script
        If it is a python script.

        e.g. if localpath = /home/user/scripts

        and path is /home/user/scripts/foo/bar.py

        returns foo/bar.py

        If it is not a python script return None.

        e.g.  if path = ./.~c9_invoke_LSdZQw.py

        then returns None
        """
        filename = path.rsplit('/', 1)[1]
        if filename.startswith('.') or not filename.endswith('.py'):
            return None

        return path[len(self.localpath):]


@click.command()
@click.option('--recursive/--no-recursive', default=False, type=bool, help='Recursively watch subdirectories')
@click.option('--verbose/--no-verbose', default=False)
@click.argument('srcdb')
@click.argument('localpath', default='.', type=click.Path(exists=True))
def run(srcdb: str, localpath: str, recursive: bool, verbose: bool):
    """
    Synchronize code from local filesystem to remote source

    Example:


    """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    CodeSync(srcdb, localpath, recursive).watch()


if __name__ == '__main__':
    run()
