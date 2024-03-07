# setup py file for the package

from setuptools import setup, find_packages

# get requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='wsc_interview',
    version='0.0.1',
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.10',
    description='Home assignment for WSC interview.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Omer Nagar'
)
