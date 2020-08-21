from .base import BaseDB
import redis


class RedisDB(BaseDB):

    def __init__(self, uri: str):
        host, port = uri.split('://', 1)[1].split(':')
        print((host, port))
        self.connection = redis.Redis(host=host, port=port)

    def get_raw(self, key):
        return self.connection.get(key)

    def set_raw(self, key, value):
        self.connection.set(key, value)
