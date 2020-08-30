from .base import BaseDB
import requests


class HttpDB(BaseDB):
    """
    HTTP implementation of KYDB.

    Currently supports read only

    However if for example the HTTP query is requesting from a static web host
    on an S3 bucket. Then the s3 KYDB can write to it and then reading can
    be done with this same class.
    """

    def __init__(self, url: str):
        super().__init__(url)

    def get_raw(self, key: str):
        r = requests.get('{}://{}{}'.format(self.db_type, self.db_name,  key))
        if not r.ok:
            raise KeyError(key)

        return bytes.fromhex(r.text)
