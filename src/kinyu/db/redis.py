from .base import BaseDB
import redis


class RedisDB(BaseDB):

    def __init__(self, url: str):
        super().__init__(url)

        self.connection = redis.Redis(
            **self._get_connection_kwargs(self.db_name))

    @staticmethod
    def _get_connection_kwargs(db_name: str):
        if ':' in db_name:
            host, port = db_name.split(':')
            kwargs = {
                'host': host,
                'port': int(port, 10)
            }
        else:
            kwargs = {
                'host': db_name
            }

        return kwargs

    def get_raw(self, key: str):
        res = self.connection.get(key)
        if not res:
            raise KeyError(key)

        return res

    def set_raw(self, key: str, value):
        self.connection.set(key, value)

    def delete_raw(self, key: str):
        self.connection.delete(key)
