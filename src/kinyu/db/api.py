from .config import DB_MODULES
from .base import BaseDB, IDB
import importlib
from typing import Tuple


class UnionDB:
    def __init__(self, dbs: Tuple[BaseDB]):
        self.dbs = dbs

    def exists(self, key: str):
        return any(db.exists(key) for db in self.dbs)

    def __getitem__(self, key: str):
        for db in self.dbs:
            try:
                return db[key]
            except KeyError:
                pass

        raise KeyError(key)

    def __setitem__(self, key, value):
        print(f'setting {key} to {value}')
        print(f'db = {self.dbs[0]}')
        self.dbs[0][key] = value


class DBFactor:
    def __init__(self):
        self._db_cache = {}

    def connect(self, url: str) -> IDB:
        dbs = [self._connect(x) for x in url.split(';')]
        if len(dbs) == 1:
            return dbs[0]
        return UnionDB(dbs)

    def _connect(self, url: str) -> BaseDB:
        if url not in self._db_cache:
            db_cls = self._resolve_db_class(url)
            self._db_cache[url] = db_cls(url)

        return self._db_cache[url]

    def _resolve_db_class(self, url: str):
        db_type = url.split(':', 1)[0]
        assert db_type in DB_MODULES, \
            '{} is not one of the valid db types: {}'.format(
                db_type,
                list(iter(DB_MODULES.keys())))

        class_name = DB_MODULES[db_type]
        module_path = __name__.rsplit('.', 1)[0] + '.' + db_type
        m = importlib.import_module(module_path)
        return getattr(m, class_name)


kydb = DBFactor()
