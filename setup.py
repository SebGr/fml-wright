import setuptools
from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

install_dependencies = [
    "pandas==1.0.3",
    "rtree==0.9.4",
    "geopandas==0.7.0",
    "descartes==1.1.0",
    "tqdm==4.44",
    "numpy==1.18.2",
    "opencv-python==4.2.0.32",
    "tensorflow==2.3.1",
    "tensorflow-addons==0.10.0",
    "networkx==2.4",
    "pyyaml==5.3.1",
    "joblib==0.16.0",
]

setuptools.setup(
    name="fmlwright",
    version="0.1.0",
    author="Sebastiaan",
    author_email="Sebastiaan",
    description="Generate building plans at various stages",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=install_dependencies,
    packages=find_packages(),
)
