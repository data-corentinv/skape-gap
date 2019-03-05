![alt text](https://img.shields.io/badge/python-3.5-blue.svg)

skape-gap
===========

skape-gap is a Python module based on time series modeling and machine learning (trees).
See the  <a href="./AUTHORS.rst">AUTHORS.rst</a> file for a list of contributors.

Installation
------------

This package is not available on <a href="https://pypi.org/">PyPI</a> yet. To install it
clone git repository and install the package with a symlink :

```
pip install -e .
```

Usage
------
To use prophet (time serie part) and decision tree (machine learning part) modeling.
```
from skgap import SkapeGap

model = SkapeGap(time_series='prophet',
                 machine_learning='DT',

                yearly_seasonality = True,
                weekly_seasonality = False,
                daily_seasonality = False,
                seasonality_prior_scale = 0.1,

                max_depth_dt=5)

model.fit(date, X_train, y_train)
model.predict(X_test, 10)
```

Demo
-----
See notebooks directory for a demo of skgap.
