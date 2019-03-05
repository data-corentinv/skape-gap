# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 2018

@author: corentinV
@last update: 28.11.2018
@description: construction de modele
"""

# ======================== liste classes  ==================================

# SkapeGap : model combines 2 parts (time series and machine learning)

# ========================= import ==================================

from sklearn.base import BaseEstimator
from .ts_part import model_part1
from .ml_part import model_part2

import logging as log

class SkapeGap(model_part1, model_part2, BaseEstimator):
    """

    Parameters
    ----------
        PART 1 : time series paramters (cf. ts_part.py)

        yearly_seasonality: bool (default True)
            param prophet algo
        weekly_seasonality: bool (default True)
            param prophet
        daily_seasonality: bool (default False)
            param prophet
        seasonality_prior_scale: float (default 0.1)
            param prophet
        seasonal_periods: int (default 7)
            param exponential smoothing
        trend: str (default additive)
            param exponential smoothing
        seasonal: str (default additive)
            param exponential smoothing
        put_zeros: bool (default: True)
            put zeros negatives forecasting

        PART 2 : machine learning parameters (cf. ml_part.py)

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

        time_series: str (default 'prophet')
            choix de la methode (2 choices : 'prophet' or 'exponential smoothing')
        machine_learning: str (default 'DT')
            choix de la methode de machine learning ('DT' pour decision tree ou 'RF' pour random forest)

    Attributes
    ----------
        model_prophet: object
            prophet model used
        model_exp_smoothing: object
            exponential smoothing model used
        mdel_dt: object
            decision tree model used
        model_rf: object
            random forest model used
        preprocessing_saved: dict
            preprocessing encoding saved

    Functions
    ---------
        PART 1: cf model_part1
            fit_prophet: fit model prophet
            predict_prophet: make prediction with prophet
            fit_exp_smoothing: fit model exponential smoothing
            predict_exp_smoothing: fit model exponential smoothing

        PART 2: cf model_part2
            fit_dt : fit decision tree model
            predict_dt : make prediction with decision tree model
            fit_rf : fit random forest model
            predict_rf : make prediction with random forest model

        fit : fit both models
        predict : make prediction with both models
        score: return the coefficient of determination R^2 of the prediction

    """
    def __init__(self,
            yearly_seasonality = True,
            weekly_seasonality = True,
            daily_seasonality = False,
            seasonality_prior_scale = 0.1,
            seasonal_periods = 7,
            trend = 'additive',
            seasonal = 'additive',
            put_zeros=True,

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

            time_series = 'prophet',
            machine_learning = 'DT',

            random_state=42):

        self.time_series = time_series
        self.machine_learning = machine_learning

        model_part1.__init__(self,
                    yearly_seasonality = yearly_seasonality,
                    weekly_seasonality = weekly_seasonality,
                    daily_seasonality = daily_seasonality,
                    seasonality_prior_scale = seasonality_prior_scale,
                    seasonal_periods = seasonal_periods,
                    trend = trend,
                    seasonal = seasonal,
                    put_zeros=put_zeros)

        model_part2.__init__(self, criterion_dt=criterion_dt,
                    splitter_dt=splitter_dt,
                    max_depth_dt=max_depth_dt,
                    min_samples_split_dt=min_samples_split_dt,
                    min_samples_leaf_dt=min_samples_leaf_dt,
                    max_features_dt=max_features_dt,
                    n_estimators_rf=n_estimators_rf,
                    criterion_rf=criterion_rf,
                    max_depth_rf=max_depth_rf,
                    min_samples_split_rf=min_samples_split_rf,
                    min_samples_leaf_rf=min_samples_leaf_rf,
                    max_features_rf=max_features_rf,
                    random_state=random_state)

    def fit(self, date, X, y):
        """ fit time serie and machine learning model

        Parameters
        ----------
            date: column pandas, list
                list of dates
            X: pandas DataFrame
                data frame contains features used in machine learning model
            y: columns pandas, list
                output to predict

        Returns
        ----------
            self: object
                model fitted
        """
        # fit model part 1
        if self.time_series == 'prophet':
            self.fit_prophet(date, y)
        else:
            # to do random forest
            log.error('TO DO: exponential smoothing not implemented yet')
            return 1

        pred = self.predict_prophet(0)
        deviation = y - pred.yhat

        # fit model part 2
        if self.machine_learning=='DT':
            self.fit_dt(X, deviation)
        elif self.machine_learning == 'RF':
            self.fit_rf(X, deviation)
        else:
            #to do random forest model
            log.error('Model unknown')
            return 1


    def predict(self, X, period=0):
        """ make prediction

        Parameters
        ----------
            X: pandas dataframe
                features used in machine learning part
            period: int
                number of day to predict

        Returns
        ----------
            self: object
                model fitted
        """
        if self.time_series == 'prophet':
            if period >= 0:
                pred_part1 = self.predict_prophet(period).yhat[-period:]
            else:
                log.error('Period cannot be negative')
                return 1
        else:
            #to do random forest model
            log.error('TO DO: exponential smoothing not implemented yet')
            return 1

        if self.machine_learning == 'DT':
            pred_part2 = self.predict_dt(X)
        elif self.machine_learning == 'RF':
            pred_part2 = self.predict_rf(X)
        else:
            #to do random forest model
            log.error('Model unkown')
            return 1

        pred = pred_part1 + pred_part2

        if self.put_zeros:
            pred[pred < 0 ] = 0

        return pred
        
    def score(self, X, y):
        """ return the coefficient of determination R^2 of the prediction

        Parameters
        ----------
            X: array-like, shape = (n_samples, n_features)
                test samples
            y: array-like, shape = (n_samples) or (n_samples, n_outputs)
                output
        Returns
        ----------
            self: object
                model fitted
        """
        if hasattr(self, 'period'):
            pred = self.predict(X, self.period)
            return r2_score(y, pred)
        else:
            log.error('period unhnown')
            return 1
