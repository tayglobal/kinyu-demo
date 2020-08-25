from abc import ABC
import pickle
import importlib


class IDB(ABC):
    def __getitem__(self, key: str):
        raise NotImplementedError()
        
    def __setitem__(self, key: str, value):
        raise NotImplementedError()

class BaseDB(IDB):
    def __init__(self, url: str):
        self.url = url
        self._cache = {}

    def _serialise(self, obj):
        return pickle.dumps(obj)

    def _deserialise(self, obj):
        return pickle.loads(obj)
        
    def exists(self, key) -> bool:
        try:
            self[key]
            return True
        except:
            return False

    def __getitem__(self, key: str):
        return self.read(key)
        
    def refresh(self, key=None):
        if key:
            del self._cache[key]
        else:
            self._cache = {} 
            
    def read(self, key: str, reload=False):
        try:
            res = None if reload else self._cache.get(key)
            if not res:
                res = self._deserialise(self.get_raw(key))
                
            self._cache[key] = res
            return res
                
        except:
            raise KeyError(key)

    def __setitem__(self, key: str, value):
        self._cache[key] = value
        self.set_raw(key, self._serialise(value))

    def get_raw(self, key):
        raise NotImplementedError()

    def set_raw(self, key, value):
        raise NotImplementedError()
        
    def new(self, class_path: str, db_path: str, **kwargs):
        module_path, class_name = class_path.rsplit('.', 1)
        m = importlib.import_module(module_path)
        cls = getattr(m, class_name)
        return cls(self, db_path, **kwargs)
        
    def __repr__(self):
        return f'<{type(self).__name__} {self.url}>'
        
