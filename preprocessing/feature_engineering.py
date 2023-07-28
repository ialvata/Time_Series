import pandas as pd
from  abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from enum import IntEnum
from datetime import datetime

class TrigSeason(IntEnum):
    """
    Number of seconds in a cycle
    """
    DAY = 24*60*60
    WEEK = 7*DAY
    MONTH = 30*DAY
    YEAR = 365*DAY

class FeatureEngineering(ABC):
    def __init__(self, dataframe:pd.DataFrame):
        self.dataframe = dataframe

    def add_lags(self, n_lags:int|list[int], drop_cols:list)-> pd.DataFrame:
        """
        Each row of df must be a separate time point, which will be transformed
        into a lag. This function will transform a matrix of dim -> n_samples x n_columns
        into a matrix of dim -> (n_samples-n_lags) x (n_columns*n_lags)
        Attention
        ---------
        For use of lagged values as input, we should favor use of already existing rows 
        instead of new columns creation as is done here. The use of already existing rows is
        more efficient when it comes to memory consumption.
        """
        if isinstance(n_lags,int):
            lags = range(1,n_lags+1)
        elif isinstance(n_lags,list):
            lags = n_lags
        appended_lags = []
        for lag in lags: 
            lag_df= self.dataframe.shift(lag).drop(columns=drop_cols)
            lag_df.columns=[x+"_lag_"+str(lag) for x in lag_df.columns]
            appended_lags.append(lag_df)
        return (
            pd.concat(appended_lags, axis=1)\
                .dropna()\
                # droping the empty cells caused by shifting
                .reset_index(drop=True)\
                # resetting index so that .loc[0] gives 1st row
        )
    
    def add_trig_season(self, seasonality:TrigSeason, 
                        time_column: pd.DatetimeIndex | None = None):
        """
        This method will encode time as a usable feature. The main difference between 
        using columns with month, day and other time related quantities, is that these
        trignometric features will be in the range [0,1], which will help when using 
        models which require scalling the features to a certain range.

        The way it works is that we transform the date into seconds, and then apply sine,
        and cosine functions on it, according to how many seconds the cycle has.
        A day has 24*60*60 seconds.
        A week has 7*24*60*60 seconds.
        A month has 30*24*60*60 seconds. (Disregarding 28/29/31 days months)
        A year has 365*24*60*60 seconds. (Disregarding 366 days years)
        We may have some problems if we accumulate many months different from 30 days, or
        years different from 365 days.

        Parameters:
        -----------

        `time_column`: pd.Series | pd.DataFrame
            This input variable is an array like var that contains the time/date for each
            observation in self.dataframe.
        """
        if time_column is None:
            if not isinstance(self.dataframe.index,pd.DatetimeIndex):
                raise Exception("Please convert DataFrame Index to a DatetimeIndex!")
            time_column = self.dataframe.index
            timestamp_seconds = time_column.map(datetime.timestamp)
            sin_name = f"{seasonality.name}_sin_season"
            cos_name = f"{seasonality.name}_cos_season"
            self.dataframe[sin_name] = (
                np.sin(timestamp_seconds * (2 * np.pi / seasonality))
            )
            self.dataframe[cos_name] = (
                np.cos(timestamp_seconds * (2 * np.pi / seasonality))
            )