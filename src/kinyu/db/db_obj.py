from .base import BaseDB
from abc import ABC


class DbObj(ABC):
    def __init__(self, db: BaseDB, path: str = None, **kwargs):
        self.db = db
        self.path = path
        self.init(**kwargs)

    def init(self, **kwargs):
        raise NotImplementedError()

    def put(self):
        self.db[self.path] = self

    def __state__(self):
        return {k: v for k, v in self.__dict__ if k != 'db'}
