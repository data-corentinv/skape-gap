# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 2018

@author: corentinV
@last update: 28.11.2018
@description: construction de modele part 2
"""

import logging as log

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

import datetime as dt
import numpy as np
import pandas as pd

class model_part2():
    """ construction d'un model v0 part 2 (machine learning) (2 choix: decision tree ou random forest)

    TO DO
    -----
        ajouter l'information num_ent/num_ett

    Parameters
    ----------
        criterion_dt: str (default 'mse')
            decision tree regressor parameters
        splitter_dt: str (default 'best')
            decision tree regressor parameter
        max_depth_dt: int (default None)
            decision tree regressor parameter
        min_samples_leaf_dt: int (default 1)
            decision tree regressor parameter
        min_samples_split_dt: int (default 2)
            decision tree regressor parameter
        max_features_dt: int (default None)
            decision tree regressor parameter
        n_estimators_rf: int (default 'warn')
            random forest regressor parameter
        criterion_rf: str (default 'mse')
            random forest regressor parameter
        max_depth_rf: int (default None)
            random forest regressor parameter
        min_samples_split_rf: int (default 2)
            random forest regressor parameter
        min_samples_leaf_rf: int (default 1)
            random forest regressor parameter
        max_features_rf: int (default 'auto')
            random forest regressor parameter
        random_state: int (default 42)
            decision tree regressor parameter

    Attributes
    ----------
        model_dt: object
            decision tree model used
        model_rf: object
            random forest model used

    Functions
    ---------
        fit_dt : fit decision tree model
        predict_dt : make prediction with decision tree model
        fit_rf : fit random forest model
        predict_rf : make prediction with random forest model

    """
    def __init__(self,
            criterion_dt='mse',
            splitter_dt='best',
            max_depth_dt=None,
            min_samples_split_dt=2,
            min_samples_leaf_dt=1,
            max_features_dt=None,

            n_estimators_rf='warn',
            criterion_rf='mse',
            max_depth_rf=None,
            min_samples_split_rf=2,
            min_samples_leaf_rf=1,
            max_features_rf='auto',

            random_state=42):

        # params decision tree regressor
        self.criterion_dt = criterion_dt
        self.splitter_dt = splitter_dt
        self.max_depth_dt = max_depth_dt
        self.min_samples_split_dt = min_samples_split_dt
        self.min_samples_leaf_dt = min_samples_leaf_dt
        self.max_features_dt = max_features_dt

        self.random_state = random_state

        #params random forest regressor
        self.n_estimators_rf = n_estimators_rf
        self.criterion_rf = criterion_rf
        self.max_depth_rf = max_depth_rf
        self.min_samples_split_rf = min_samples_split_rf
        self.min_samples_leaf_rf = min_samples_leaf_rf
        self.max_features_rf = max_features_rf

    def fit_dt(self, X, y):
        """ fit model decision tree

        Parameters
        ----------
            X: pandas DataFrame
                dataset with insteresting features
            y: column pandas DataFrame
                output to predict

        Returns
        ----------
            self: object
                model fitted
        """

        clf = DecisionTreeRegressor(criterion=self.criterion_dt,
                    splitter=self.splitter_dt,
                    max_depth=self.max_depth_dt,
                    min_samples_split=self.min_samples_split_dt,
                    min_samples_leaf=self.min_samples_leaf_dt,
                    max_features=self.max_features_dt,
                    random_state=self.random_state)
        self.model_dt = clf
        self.model_dt.fit(X,y)


    def predict_dt(self, X):
        """ make predictions with decision tree model

        Parameters
        ----------
            periode_prediction: int
                number of day to predict

        Returns
        ----------
            forecast: pandas dataframe
                prediction with lower, upper prediction and trend etc.
                (cf. prophet documentation for more details)
        """
        clf = self.model_dt
        if hasattr(self, 'model_dt'):
            pred = clf.predict(X)
            return pred
        else:
            log.error('fit random forest model before make predictions')
            return 1


    def fit_rf(self, X, y):
        """ make predictions with random forest model

        Parameters
        ----------
            y: list of float
                output to predict

        Returns
        ----------
            self: object
                model fitted
        """
        clf = RandomForestRegressor(n_estimators=self.n_estimators_rf,
                            criterion=self.criterion_rf,
                            max_depth=self.max_depth_rf,
                            min_samples_split=self.min_samples_split_rf,
                            min_samples_leaf=self.min_samples_leaf_rf,
                            max_features=self.max_features_rf,
                            random_state=self.random_state)

        self.model_rf = clf
        clf.fit(X,y)


    def predict_rf(self, X):
        """ make predictions with exponential smoothing

        Parameters
        ----------
            periode_prediction: int
                number of day to predict

        Returns
        ----------
            forecast: list
                list of forecast
        """
        clf = self.model_rf
        if hasattr(self, 'model_rf'):
            pred = clf.predict(X)
            return pred
        else:
            log.error('fit random forest model before make predictions')
            return 1
