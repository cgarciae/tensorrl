from setuptools import find_packages
from setuptools import setup



setup(
    name = 'TensorRL',
    version = '0.0.1',
    author = 'Cristian Garcia',
    author_email = 'cgarcia.e88@gmail.com',
    packages = find_packages(),
    url = 'https://github.com/cgarciae/tensorrl/',
    license = 'LICENSE',
    description = '',
    long_description = open('README.md').read(),
    install_requires = [
        "keras-rl >= 0.4.2",
    ],
)