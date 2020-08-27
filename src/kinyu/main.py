import click
import importlib
import json
from kinyu.rimport.api import RemoteImporter


@click.command()
@click.option('--srcdb', help='The source db')
@click.option('--entry', default='main', help='The entry function')
@click.option('--args', default='{}', help='Arguments to be passed into entry function in JSON format')
@click.argument('module_path')
def run(srcdb, entry, args, module_path):
    ri = RemoteImporter(srcdb)
    ri.install()
    m = importlib.import_module(module_path)
    getattr(m, entry)(**json.loads(args))


if __name__ == '__main__':
    run()
