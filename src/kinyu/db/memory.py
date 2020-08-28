from .base import BaseDB


class MemoryDB(BaseDB):
    __cache = {}

    def __init__(self, url: str):
        super().__init__(url)
        self.__cache[self.db_name] = {}

    def get_raw(self, key):
        return self.__cache[self.db_name][key]

    def set_raw(self, key, value):
        self.__cache[self.db_name][key] = value

    def delete_raw(self, key: str):
        del self.__cache[self.db_name][key]
