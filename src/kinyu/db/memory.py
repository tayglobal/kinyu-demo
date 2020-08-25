from .base import BaseDB


class MemoryDB(BaseDB):
    __cache = {} 

    def __init__(self, url: str):
        super().__init__(url)
        self.cache_name = url.split('://', 1)[1]
        self.__cache[self.cache_name] = {}

    def get_raw(self, key):
        return self.__cache[self.cache_name][key]

    def set_raw(self, key, value):
        self.__cache[self.cache_name][key] = value
