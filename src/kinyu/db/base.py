from abc import ABC
import pickle


class BaseDB(ABC):
    def __init__(self, url: str):
        raise NotImplementedError()

    def _serialise(self, obj):
        return pickle.dumps(obj)

    def _deserialise(self, obj):
        return pickle.loads(obj)

    def __getitem__(self, key):
        return self._deserialise(self.get_raw(key))

    def __setitem__(self, key, value):
        return self.set_raw(key, self._serialise(value))

    def get_raw(self, key):
        raise NotImplementedError()

    def set_raw(self, key, value):
        raise NotImplementedError()
