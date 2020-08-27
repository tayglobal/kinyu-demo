import click
import importlib
from kinyu.rimport.api import RemoteImporter


@click.command()
@click.option('--srcdb', help='The source db')
@click.argument('module_path')
def run(srcdb, module_path):
    ri = RemoteImporter(srcdb)
    ri.add_script("unittest/main_module.py", '''
def main():
    print("Hello World")
''')
    ri.install()
    m = importlib.import_module(module_path)
    m.main()


if __name__ == '__main__':
    run()
