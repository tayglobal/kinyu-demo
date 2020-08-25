from .base import BaseDB
import s3fs


class S3DB(BaseDB):

    def __init__(self, url: str):
        super().__init__(url)
        self.prefix = url
        if not self.prefix.endswith('/'):
            self.prefix += '/'

        self.fs = s3fs.S3FileSystem()

    def get_raw(self, key):
        return self.fs.open(
            self._get_full_path(key), 'rb').read()

    def set_raw(self, key, value):
        with self.fs.open(self._get_full_path(key), 'wb') as f:
            f.write(value)

    def _get_full_path(self, key):
        return self.prefix + key
