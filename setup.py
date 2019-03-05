import sys
from setuptools import setup, find_packages

try:
    import numpy
except ImportError:
    print('numpy is required during installation')
    sys.exit(1)

setup(name='skape-gap',
    version='1.0.0',
    decription='Combination between Time Series and Machine Learning modeling',
    url='',
    author='see AUTHORS.rst',
    packages=find_packages(),
    author_email='corentin.vasseur@leroymerlin.fr',
    keywords=['time series', 'machine learning', 'prevision'],
    install_requires=['numpy>=1.10.4',
                    'scikit-learn>=0.17.1',
                    'pandas>=0.18.1',
                    'fbprophet==0.3', #[Error]: version 0.4 pose pblm
                    'statsmodels>=0.9.0'])
