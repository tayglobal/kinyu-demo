from .config import DB_MODULES
import importlib


class DBFactor:
    def __init__(self):
        self._db_cache = {}

    def connect(self, url) -> 'BaseDB':
        if url not in self._db_cache:
            db_cls = self._resolve_db_class(url)
            self._db_cache[url] = db_cls(url)

        return self._db_cache[url]

    def _resolve_db_class(self, url: str):
        db_type = url.split(':', 1)[0]
        assert db_type in DB_MODULES, '{} is not one of the valid db types: {}'.format(
            db_type,
            list(iter(DB_MODULES.keys())))

        class_name = DB_MODULES[db_type]
        module_path = __name__.rsplit('.', 1)[0] + '.' + db_type
        m = importlib.import_module(module_path)
        return getattr(m, class_name)


kydb = DBFactor()
