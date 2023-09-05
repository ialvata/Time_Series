import pandas as pd
from  abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from enum import IntEnum
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler,MaxAbsScaler

class TrigSeason(IntEnum):
    """
    Number of seconds in a cycle
    """
    DAY = 24*60*60
    WEEK = 7*DAY
    MONTH = 30*DAY
    YEAR = 365*DAY

class BaseScaler:
    pass

class FeatureEngineering(ABC):
    def __init__(self, dataframe:pd.DataFrame, 
                 scaler:MinMaxScaler | StandardScaler | MaxAbsScaler | None = None):
        self._dataframe = dataframe
        self.scaler = scaler
    
    @property
    def dataframe(self) -> pd.DataFrame:
        return self._dataframe.dropna()

    def add_lags(self, 
                 n_lags:int|list[int], columns:list,
                 return_output: bool = False)-> pd.DataFrame | None:
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
            lag_df= self._dataframe.shift(lag).drop(columns=columns)
            lag_df.columns=[x+"_lag_"+str(lag) for x in lag_df.columns]
            appended_lags.append(lag_df)
        if return_output:
            return (
                pd.concat(appended_lags, axis=1)\
                    .dropna()\
                    # droping the empty cells caused by shifting
                    .reset_index(drop=True)\
                    # resetting index so that .loc[0] gives 1st row
            )
        else:
            full_list_df = [self._dataframe] + appended_lags
            self._dataframe = pd.concat(full_list_df, axis=1)
    
    def add_rolling_features(self,
                             columns:list[str],
                             rolling_periods:int|list[int],
                             return_output: bool = False)-> pd.DataFrame | None:
        """
        Parameters
        ----------
        `columns:list[str]`
            A list with the columns names(`str`) for which we want to create the rolling\
                features.

        `rolling_periods:int|list[int]`
            \tIf `rolling_periods` is int, then we internally create a list from 1 to \
            `rolling_periods`+1, and use each element as a different period to create a \
                feature.
            \tIf `rolling_periods` is a list, then we just use the periods explicitely 
            in the list.
        
        TODO:
            Add other aggregation functions besides `mean`, such as `std`, `min`, and `max`.

        """
        if isinstance(rolling_periods,int):
            periods = range(1,rolling_periods+1)
        elif isinstance(rolling_periods,list):
            periods = rolling_periods
        appended_lags = []
        for period in periods: 
            rolling_df = self._dataframe[columns]
            rolling_df.columns=[x+"_roll_avg_"+str(period) for x in columns]   
            rolling_df = rolling_df.rolling(window = period).mean()
            appended_lags.append(rolling_df)
        if return_output:
            return (
                pd.concat(appended_lags, axis=1)\
                    .dropna()\
                    # droping the empty cells caused by shifting
                    .reset_index(drop=True)\
                    # resetting index so that .loc[0] gives 1st row
            )
        else:
            full_list_df = [self._dataframe] + appended_lags
            self._dataframe = pd.concat(full_list_df, axis=1)


    def scale_features(self)-> None:
        """
        This method will scale *all* features in the self.dataframe
        """
        if self.scaler is None:
            self.scaler = MinMaxScaler()
            self.scaler.fit(self.dataframe)
        self.scaler.transform(self.dataframe)

    def add_trig_season(self, seasonality:TrigSeason, 
                        time_column: pd.DatetimeIndex | None = None)-> None:
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
            self._dataframe[sin_name] = (
                np.sin(timestamp_seconds * (2 * np.pi / seasonality))
            )
            self._dataframe[cos_name] = (
                np.cos(timestamp_seconds * (2 * np.pi / seasonality))
            )