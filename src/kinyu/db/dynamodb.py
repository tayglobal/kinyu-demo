from .base import BaseDB
from boto3.dynamodb.conditions import Key
import boto3


class DynamoDB(BaseDB):

    def __init__(self, url: str):
        super().__init__(url)
        dynamodb = boto3.resource('dynamodb')
        self.table = dynamodb.Table(self.db_name)

    def get_raw(self, key):
        items = self.table.query(
            KeyConditionExpression=Key('path').eq(key))['Items']

        if not items:
            raise KeyError(key)

        return items[0]['contents'].value

    def set_raw(self, key, value):
        self.table.put_item(Item={
            'path': key,
            'contents': value
        })

    def delete_raw(self, key: str):
        self.table.delete_item(Key={
            'path': key,
        })
