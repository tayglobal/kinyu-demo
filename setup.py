from setuptools import setup, find_packages
from setuptools_rust import RustExtension

setup(
    name="kinyu",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    rust_extensions=[
        # RustExtension("kinyu_graph", "src/kinyu/graph/Cargo.toml", debug=False),
        RustExtension("kinyu_historical", "src/kinyu/vol/historical/Cargo.toml", debug=False),
        RustExtension("kinyu.warrants", "src/kinyu/warrants/Cargo.toml", debug=False),
    ],
    # rust extensions are not zip safe, just like C-extensions.
    zip_safe=False,
    include_package_data=True,
)