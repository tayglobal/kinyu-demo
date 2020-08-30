from abc import ABC
import pickle
import importlib
from typing import Tuple


class IDB(ABC):
    def __getitem__(self, key: str):
        raise NotImplementedError()

    def __setitem__(self, key: str, value):
        raise NotImplementedError()

    def delete(self, key: str):
        raise NotImplementedError()


class BaseDB(IDB):
    def __init__(self, url: str):
        self.db_type = url.split(':', 1)[0]
        self.db_name, self.base_path = self._get_name_and_basepath(url)
        self.url = url
        self._cache = {}

    @staticmethod
    def _get_name_and_basepath(url: str) -> Tuple[str, str]:
        base_path = '/'
        parts = url.split('/', 3)
        num_parts = len(parts)
        assert num_parts in (3, 4)

        db_name = parts[2]
        if len(parts) == 4:
            base_path += parts[-1]

        if base_path[-1] != '/':
            base_path += '/'

        return db_name, base_path

    def _get_full_path(self, key: str) -> str:
        if key.startswith('/'):
            key = key[1:]

        return self.base_path + key

    def _serialise(self, obj):
        return pickle.dumps(obj)

    def _deserialise(self, obj):
        return pickle.loads(obj)

    def exists(self, key) -> bool:
        try:
            self[key]
            return True
        except KeyError:
            return False

    def __getitem__(self, key: str):
        return self.read(key)

    def refresh(self, key=None):
        if key:
            del self._cache[key]
        else:
            self._cache = {}

    def read(self, key: str, reload=False):
        path = self._get_full_path(key)
        res = None if reload else self._cache.get(path)
        if not res:
            res = self._deserialise(self.get_raw(path))

        self._cache[key] = res
        return res

    def __setitem__(self, key: str, value):
        self._cache[key] = value
        self.set_raw(self._get_full_path(key), self._serialise(value))

    def get_raw(self, key: str):
        raise NotImplementedError()

    def set_raw(self, key: str, value):
        raise NotImplementedError()

    def delete_raw(self, key: str):
        raise NotImplementedError()

    def delete(self, key: str):
        del self._cache[key]
        self.delete_raw(key)

    def new(self, class_path: str, db_path: str, **kwargs):
        module_path, class_name = class_path.rsplit('.', 1)
        m = importlib.import_module(module_path)
        cls = getattr(m, class_name)
        return cls(self, db_path, **kwargs)

    def __repr__(self):
        return f'<{type(self).__name__} {self.url}>'
