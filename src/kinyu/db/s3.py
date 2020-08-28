from .base import BaseDB
import s3fs


class S3DB(BaseDB):

    def __init__(self, url: str):
        super().__init__(url)
        self.prefix = url
        if not self.prefix.endswith('/'):
            self.prefix += '/'

        self.fs = s3fs.S3FileSystem()

    def get_raw(self, key: str):
        return self.fs.open(
            self._get_s3_path(key), 'rb').read()

    def set_raw(self, key: str, value):
        with self.fs.open(self._get_s3_path(key), 'wb') as f:
            f.write(value)

    def _get_s3_path(self, key: str):
        return 's3://' + self.db_name + self._get_full_path(key)

    def delete_raw(self, key: str):
        self.fs.rm(self._get_s3_path(key))
