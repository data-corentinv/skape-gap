# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 2018

@author: corentinV
@last update: 28.11.2018
@description: construction de modele
"""

import logging as log

from fbprophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import datetime as dt
import numpy as np
import pandas as pd

# Rmq : mise a 0 des valeurs négatives (a faire avant le rassemblement des familles logistiques ou apres ? check)

class model_part1():
    """ construction d'un model v0 part 1 (time series) (2 choix: prophet ou exponential smoothing)

    TO DO
    -----
        ajouter l'information num_ent/num_ett

    Parameters
    ----------

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

    Attributes
    ----------
        model_prophet: object
            prophet model used
        model_exp_smoothing: object
            exponential smoothing model used

    Functions
    ---------
        fit_prophet: fit model prophet
        predict_prophet: make prediction with prophet
        fit_exp_smoothing: fit model exponential smoothing
        predict_exp_smoothing: fit model exponential smoothing

    """
    def __init__(self,
                yearly_seasonality = True,
                weekly_seasonality = True,
                daily_seasonality = False,
                seasonality_prior_scale = 0.1,
                seasonal_periods = 7,
                trend = 'additive',
                seasonal = 'additive',
                put_zeros=True):

        self.put_zeros = put_zeros

        # params prophet
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_prior_scale = seasonality_prior_scale

        # params exponential smoothing
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.seasonal = seasonal

    def fit_prophet(self, date, y):
        """ fit model prophet

        Parameters
        ----------
            date: column pandas DataFrame contains dates
            y: column pandas Dataframe contrains sortie

        Returns
        ----------
            self: object
                model fitted
        """
        train = pd.DataFrame(data={'ds': date, 'y': y})

        m = Prophet(yearly_seasonality=self.yearly_seasonality, \
                    daily_seasonality=self.daily_seasonality, \
                    weekly_seasonality = self.weekly_seasonality, \
                    seasonality_prior_scale=self.seasonality_prior_scale)
        m.fit(train)
        self.model_prophet = m

    def predict_prophet(self, periode_prediction):
        """ make predictions with prophet model

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
        if not hasattr(self, 'model_prophet'):
            log.error('[Error] fit model before predict')
            return 1
        else:
            m = self.model_prophet
            future = m.make_future_dataframe(periods=periode_prediction)
            forecast = m.predict(future)

            if self.put_zeros:
                forecast_yhat = forecast.yhat
                forecast_yhat[forecast_yhat < 0] = 0
                forecast.yhat = forecast_yhat

            return forecast

    def fit_exp_smoothing(self, y):
        """ make predictions with exponential smoothing model

        Parameters
        ----------
            y: list of float
                output to predict

        Returns
        ----------
            self: object
                model fitted
        """

        model = ExponentialSmoothing(y, seasonal_periods=self.seasonal_periods, trend=self.trend, seasonal=self.seasonal).fit()

        self.model_exp_smoothing = model

    def predict_exp_smoothing(self, periode_prediction):
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

        if not hasattr(self, 'model_exp_smoothing'):
            log.error('[Error] exponential smoothing not fiited')
            return 1
        else:
            model = self.model_exp_smoothing
            forecast = model.forecast(periode_prediction)

            if self.put_zeros:
                forecast[forecast < 0] = 0

            return forecast
