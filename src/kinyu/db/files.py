from .base import BaseDB
import os


class FileDB(BaseDB):

    def __init__(self, url: str):
        super().__init__(url)

    def get_raw(self, key: str):
        try:
            return open(
                self._get_fs_path(key), 'rb').read()
        except FileNotFoundError:
            raise KeyError(key)

    def set_raw(self, key: str, value):
        with open(self._get_fs_path(key), 'wb') as f:
            f.write(value)

    def _get_fs_path(self, key: str):
        return '/' + self.db_name + self._get_full_path(key)

    def delete_raw(self, key: str):
        os.remove(self._get_fs_path(key))
