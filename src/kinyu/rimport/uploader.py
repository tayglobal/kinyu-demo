import click
from kinyu.rimport.api import RemoteImporter


@click.command()
@click.option('--srcdb')
@click.option('--key')
@click.argument('filepath')
def run(srcdb, key, filepath):
    ri = RemoteImporter(srcdb)
    print(f"Uploading {filepath} to {srcdb}/{key}")
    ri.add_script_from_file(key, filepath)


if __name__ == '__main__':
    run()
