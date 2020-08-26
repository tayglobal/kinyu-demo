import timeit
import sys
import importlib
import types
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
from datetime import datetime
from kinyu.db.api import kydb
from kinyu.db.base import IDB


class RemoteFinder(MetaPathFinder):
    def __init__(self, db: IDB):
        self.db = db

    def find_spec(self, fullname, path, target=None):
        key = fullname.replace('.', '/') + '.py'
        if self.db.exists(key):
            return ModuleSpec(fullname, self, origin=key)

        key = fullname.replace('.', '/') + '/__init__.py'
        
        if self.db.exists(key):
            return ModuleSpec(fullname, self, origin=key)

    def create_module(self, spec):
        """
        Module creator. Returning None causes Python to use the default module creator.
        """
        print(spec)
        name_parts = spec.name.split('.')
        module = types.ModuleType('.'.join(name_parts))
        module.__path__ = name_parts[:-1]
        module.__file__ = spec.origin
        return module

    def exec_module(self, module):
        code = self.db[module.__file__]['code']
        exec(code, module.__dict__)
        return module


class RemoteImporter:
    def __init__(self, url: str):
        self.url = url
        self.db = kydb.connect(url)
        
    def install(self):
        finder = RemoteFinder(self.db)
        sys.meta_path.append(finder)