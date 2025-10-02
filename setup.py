from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name = 'Research Chat',
    version='0.1',
    author='Ujjawal',
    packages=find_packages(),
    install_requires = requirements
)