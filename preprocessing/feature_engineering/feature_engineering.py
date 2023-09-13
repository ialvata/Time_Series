import pandas as pd
from  abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from enum import IntEnum
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler,MaxAbsScaler
from preprocessing.data_spliting.label_features import LabelFeatSet
from preprocessing.feature_engineering.feature_base import FeatureBase

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
    def __init__(self,
                 labels_names:list[str],
                 dataframe:pd.DataFrame | None = None,
                 scaler:MinMaxScaler | StandardScaler | MaxAbsScaler | None = None,
                 ):
        self._dataframe = dataframe
        self.scaler = scaler
        self.labels_names = sorted(labels_names)
        self.pipeline = []

    @property
    def dataframe(self) -> pd.DataFrame:
        if self._dataframe is not None:
            return self._dataframe.dropna()
        else:
            raise Exception(
                "No Dataframe for this FeatureEngineering Pipeline has been defined!"
            )
    
    @property
    def labels(self)-> pd.DataFrame | pd.Series:
        return self.dataframe[self.labels_names]
    
    @property
    def features(self)-> pd.DataFrame | pd.Series:
        """
        Anything which is not a label, will be considered a feature.
        """
        # if we do feature engineering, we may want to update self.dataframe after
        # TrainSet initialization. 
        # I'm not sure... Provisional pattern.
        columns_set = set(self.dataframe.columns)
        self.feature_columns = sorted(
            list(columns_set.difference(self.labels_names))
        )
        return self.dataframe[self.feature_columns]
    
    def add_to_pipeline(self, features: list[FeatureBase]) -> None:
        self.pipeline = self.pipeline + features

    def set_datasets(self, train_set: pd.DataFrame, test_set: pd.DataFrame) -> None:
        self.train_set = LabelFeatSet(train_set, labels_names=self.labels_names)
        self.test_set = LabelFeatSet(test_set, labels_names=self.labels_names)

    def create_features(self, destiny_set: str, dataframe: pd.DataFrame | None = None
    ) -> pd.DataFrame | None:
        """
        Parameters
        ----------
        `destiny_set: {"train","test", "feat_eng", "input"}`
            If "train" / "test", we send the results to train_set / test_set dataframe.

            If "feat_eng", we send the results to the dataframe initially associated to the
            FeatureEngineering instance.

            If "input", we output the inputed dataframe, expanded by the created features.
        """
        match destiny_set:
            case "train":
                resulting_df = self.train_set.dataframe
            case "test":
                resulting_df = self.test_set.dataframe
            case "feat_eng":
                if self._dataframe is None:
                    raise Exception("Missing dataframe associated to FeatEngineering!")
                resulting_df = self._dataframe
            case "input":
                if dataframe is None:
                    raise Exception("Missing dataframe input in create_features!")
                resulting_df = dataframe
            case _:
                raise Exception("Invalid option for destiny_set!")
        for feature in self.pipeline:
            feature:FeatureBase
            resulting_df = feature.create_features(resulting_df)
        match destiny_set:
            case "train":
                self.train_set.dataframe = resulting_df
            case "test":
                self.test_set.dataframe = resulting_df
            case "feat_eng":
                self._dataframe = resulting_df
            case "input":
                return resulting_df
            
            

    # def add_lags(self, 
    #              n_lags:int|list[int], columns:list,
    #              return_output: bool = False)-> pd.DataFrame | None:
    #     """
    #     Each row of df must be a separate time point, which will be transformed
    #     into a lag. This function will transform a matrix of dim -> n_samples x n_columns
    #     into a matrix of dim -> (n_samples-n_lags) x (n_columns*n_lags)
    #     Attention
    #     ---------
    #     For use of lagged values as input, we should favor use of already existing rows 
    #     instead of new columns creation as is done here. The use of already existing rows is
    #     more efficient when it comes to memory consumption.
    #     """
    #     if isinstance(n_lags,int):
    #         lags = range(1,n_lags+1)
    #     elif isinstance(n_lags,list):
    #         lags = n_lags
    #     appended_lags = []
    #     for lag in lags: 
    #         lag_df= self._dataframe.shift(lag).drop(columns=columns) #type:ignore
    #         lag_df.columns=[x+"_lag_"+str(lag) for x in lag_df.columns]
    #         appended_lags.append(lag_df)
    #     if return_output:
    #         return (
    #             pd.concat(appended_lags, axis=1)\
    #                 .dropna()\
    #                 # droping the empty cells caused by shifting
    #                 .reset_index(drop=True)\
    #                 # resetting index so that .loc[0] gives 1st row
    #         )
    #     else:
    #         full_list_df = [self._dataframe] + appended_lags
    #         self._dataframe = pd.concat(full_list_df, axis=1)
    
    # def add_rolling_features(self,
    #                          columns:list[str],
    #                          rolling_periods:int|list[int],
    #                          return_output: bool = False)-> pd.DataFrame | None:
    #     """
    #     Parameters
    #     ----------
    #     `columns:list[str]`
    #         A list with the columns names(`str`) for which we want to create the rolling\
    #             features.

    #     `rolling_periods:int|list[int]`
    #         \tIf `rolling_periods` is int, then we internally create a list from 1 to \
    #         `rolling_periods`+1, and use each element as a different period to create a \
    #             feature.
    #         \tIf `rolling_periods` is a list, then we just use the periods explicitely 
    #         in the list.
        
    #     TODO:
    #         Add other aggregation functions besides `mean`, such as `std`, `min`, and `max`.

    #     """
    #     if isinstance(rolling_periods,int):
    #         periods = range(1,rolling_periods+1)
    #     elif isinstance(rolling_periods,list):
    #         periods = rolling_periods
    #     appended_lags = []
    #     for period in periods: 
    #         rolling_df = self._dataframe[columns]
    #         rolling_df.columns=[x+"_roll_avg_"+str(period) for x in columns]   
    #         rolling_df = rolling_df.rolling(window = period).mean()
    #         appended_lags.append(rolling_df)
    #     if return_output:
    #         return (
    #             pd.concat(appended_lags, axis=1)\
    #                 .dropna()\
    #                 # droping the empty cells caused by shifting
    #                 .reset_index(drop=True)\
    #                 # resetting index so that .loc[0] gives 1st row
    #         )
    #     else:
    #         full_list_df = [self._dataframe] + appended_lags
    #         self._dataframe = pd.concat(full_list_df, axis=1)
    
    # def _calculate_fourier_terms(self,
    #     seasonal_cycle: np.ndarray, cycle_len: int, n_fourier_terms: int
    # ):
    #     """
    #     Calculates Fourier Terms which will be used as features.

    #     Parameters
    #     ----------
    #     `seasonal_cycle`:
    #         For a day-of-week cycle, we expect an iterable: 
    #             [1, 2,..., 7, 1, 2,...]
    #             Here `cycle_len` is 7.
    #         For a day-of-month cycle, we expect: 
    #             [1, 2, ..., 31, 1,...]
    #             Here `cycle_len` is 31. 
    #             Note that not all months will have the same length.
    #         For a day-of-year cycle, we expect: 
    #             [1,...,366,1,...]
    #             Here `cycle_len` is 366.
    #             Note that not all years will have the same length.
    #         For month-of-year cycle, we expect:
    #             [1, 2,..., 12, 1,...]
    #             Here `cycle_len` is 12.
    #     """
    #     sin_array = np.empty((len(seasonal_cycle), n_fourier_terms), dtype="float64")
    #     cos_array = np.empty((len(seasonal_cycle), n_fourier_terms), dtype="float64")
    #     for n in range(1, n_fourier_terms + 1):
    #         # Fourier(x) =  sum_{i=1}^{N} A_n * cos(2*pi / P * n * x) + B_n sin(2*pi/P*n*x)
    #         # seasonal_cycle array gives us the x values.
    #         sin_array[:, n - 1] = np.sin((2 * np.pi * seasonal_cycle * n) / cycle_len)
    #         cos_array[:, n - 1] = np.cos((2 * np.pi * seasonal_cycle * n) / cycle_len)
    #     return np.hstack([sin_array, cos_array])

    # def add_fourier_features(self,
    #                          seasonal_lengths:list[SeasonLength],
    #                          columns:list[str] | None = None,
    #                          num_terms:list[int] | None = None,
    #                          return_output: bool = False)-> pd.DataFrame | None:
    #     """
    #     This function will create Fourier features to represent explicitely time as a 
    #     regressor.

    #     Parameters
    #     ----------
    #     `seasonal_lengths:list[SeasonLength]`

    #     `columns:list[str]| None = None`
    #         List of columns names for which we want to build the Fourier terms
    #         If `None`, then we automatically create the Fourier terms from the DataFrame index,
    #         when it's a DateTimeIndex.

    #     `terms:list[int] | None = None`
    #         List of numbers of Fourier terms for each element of seasonal length.

    #     Note
    #     ----
    #     seasonal_lengths, n_terms, and columns should match on the element order.

    #     TODO:
    #         Implement case for SeasonLength.WEEK_OF_MONTH.
    #     """
    #     if num_terms is None:
    #             # Unless otherwise stated, we'll use 5 as the number of Fourier Terms
    #             # for all new fourier features.
    #             num_terms = [5]*len(seasonal_lengths)
    #     if not isinstance(self._dataframe.index,pd.DatetimeIndex):
    #             raise Exception(
    #                 "Please convert (internal) DataFrame Index to a DatetimeIndex!"
    #             )
    #     fourier_columns = []
    #     fourier_names = []
    #     if columns is None:
    #         for season_len,num_term in zip(seasonal_lengths,num_terms):
    #             sin_columns = [
    #                 f"sin_term_{num}_{season_len.name}" for num in range(1,num_term+1)
    #             ]
    #             cos_columns = [
    #                 f"cos_term_{num}_{season_len.name}" for num in range(1,num_term+1)
    #             ]
    #             match season_len:
    #                 case SeasonLength.DAY_OF_YEAR:
    #                     fourier_array = self._calculate_fourier_terms(
    #                         seasonal_cycle=self._dataframe.index.day_of_year.values,
    #                         cycle_len=SeasonLength.DAY_OF_YEAR,
    #                         n_fourier_terms=num_term
    #                     )
    #                 case SeasonLength.WEEK_OF_YEAR: 
    #                     fourier_array = self._calculate_fourier_terms(
    #                         seasonal_cycle = np.array(
    #                             self._dataframe.index.isocalendar().week.values
    #                             ),
    #                         cycle_len=SeasonLength.WEEK_OF_YEAR,
    #                         n_fourier_terms=num_term
    #                     )
    #                 case SeasonLength.MONTH_OF_YEAR: 
    #                     fourier_array = self._calculate_fourier_terms(
    #                         seasonal_cycle = np.array(
    #                             self._dataframe.index.month.values
    #                             ),
    #                         cycle_len=SeasonLength.MONTH_OF_YEAR,
    #                         n_fourier_terms=num_term
    #                     )
    #                 case SeasonLength.DAY_OF_MONTH: 
    #                     fourier_array = self._calculate_fourier_terms(
    #                         seasonal_cycle = np.array(
    #                             self._dataframe.index.day.values
    #                             ),
    #                         cycle_len=SeasonLength.DAY_OF_MONTH,
    #                         n_fourier_terms=num_term
    #                     )
    #                 # TODO WEEK_OF_MONTH
    #                 case SeasonLength.WEEK_OF_MONTH:
    #                     raise Exception(
    #                         "The option WEEK_OF_MONTH has not been implemented yet!"
    #                         "Please choose another option."
    #                     )
    #                 case _:
    #                     raise Exception(
    #                         "Invalid SeasonLength option! Please choose a valid one."
    #                     )
    #             fourier_columns.append(np.hstack([fourier_array]))
    #             fourier_names.append(np.hstack([sin_columns,cos_columns]))                
    #     else:
    #         for column,season_len,num_term in zip(columns,seasonal_lengths,num_terms):
    #             sin_columns = [
    #                 f"sin_term_{num}_{season_len.name}" for num in range(1,num_term+1)
    #             ]
    #             cos_columns = [
    #                 f"cos_term_{num}_{season_len.name}" for num in range(1,num_term+1)
    #             ]
    #             fourier_array = self._calculate_fourier_terms(
    #                         seasonal_cycle=np.array(self._dataframe[column]),
    #                         cycle_len=season_len,
    #                         n_fourier_terms=num_term
    #                     )
    #             fourier_columns.append(np.hstack([fourier_array]))
    #             fourier_names.append(np.hstack([sin_columns,cos_columns]))
    #     fourier_df = pd.DataFrame(np.hstack(fourier_columns),
    #                     columns = np.hstack(fourier_names),
    #                     index=self._dataframe.index)
    #     if return_output:
    #         return fourier_df
    #     else:
    #         self._dataframe = pd.concat([self._dataframe,fourier_df], axis=1)

    def scale_features(self)-> None:
        """
        This method will scale *all* features in the self.dataframe
        """
        if self.scaler is None:
            self.scaler = MinMaxScaler()
            self.scaler.fit(self.dataframe)
        self.scaler.transform(self.dataframe)

    # def add_trig_season(self, seasonality:TrigSeason, 
        #                 time_column: pd.DatetimeIndex | None = None)-> None:
        # """
        # This method will encode time as a usable feature. The main difference between 
        # using columns with month, day and other time related quantities, is that these
        # trignometric features will be in the range [0,1], which will help when using 
        # models which require scalling the features to a certain range.

        # The way it works is that we transform the date into seconds, and then apply sine,
        # and cosine functions on it, according to how many seconds the cycle has.
        # A day has 24*60*60 seconds.
        # A week has 7*24*60*60 seconds.
        # A month has 30*24*60*60 seconds. (Disregarding 28/29/31 days months)
        # A year has 365*24*60*60 seconds. (Disregarding 366 days years)
        # We may have some problems if we accumulate many months different from 30 days, or
        # years different from 365 days.

        # Parameters:
        # -----------

        # `time_column`: pd.Series | pd.DataFrame
        #     This input variable is an array like var that contains the time/date for each
        #     observation in self.dataframe.
        # """
        # if time_column is None:
        #     if not isinstance(self.dataframe.index,pd.DatetimeIndex):
        #         raise Exception("Please convert DataFrame Index to a DatetimeIndex!")
        #     time_column = self.dataframe.index
        #     timestamp_seconds = time_column.map(datetime.timestamp)
        #     sin_name = f"{seasonality.name}_sin_season"
        #     cos_name = f"{seasonality.name}_cos_season"
        #     self._dataframe[sin_name] = (
        #         np.sin(timestamp_seconds * (2 * np.pi / seasonality))
        #     )
        #     self._dataframe[cos_name] = (
        #         np.cos(timestamp_seconds * (2 * np.pi / seasonality))
        #     )