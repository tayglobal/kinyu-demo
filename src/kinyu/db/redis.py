from .base import BaseDB
import redis


class RedisDB(BaseDB):

    def __init__(self, url: str):
        super().__init__(url)
        host, port = self.db_name.split(':')
        print((host, port))
        self.connection = redis.Redis(host=host, port=port)

    def get_raw(self, key):
        return self.connection.get(key)

    def set_raw(self, key, value):
        self.connection.set(key, value)
