from setuptools import setup

setup(
    name='melvin',
    version='0.1',
    license='MIT',
    author='Jamie Quinn',
    author_email='jamiejquinn@jamiejquinn.com',
    packages=['melvin'],
    url='https://github.com/jamiejquinn/melvin.py',
    extras_require={
        'cupy': ['cupy']
    }
)
