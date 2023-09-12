import numpy as np
import pandas as pd

class RollingFeature:
    _type = "RollingFeature"
    def __init__(self,columns:list[str],
                rolling_periods:int|list[int]) -> None:
            """
            This function will create rolling time features, features containing averages of
            past values of other features.

            Parameters
            ----------
            `rolling_periods:int|list[int]`
                \tIf `rolling_periods` is int, then we internally create a list from 1 to \
                `rolling_periods`+1, and use each element as a different period to create a \
                feature.

                \tIf `rolling_periods` is a list, then we just use the periods explicitely 
                in the list.

            `columns:list[str]`
                List of columns names for which we want to build the time lags.

            Note
            ----
            This function will transform a matrix of dim -> n_samples x n_columns
            into a matrix of dim -> (n_samples-lags) x (n_columns*lags).

            TODO:
                Implement case for SeasonLength.WEEK_OF_MONTH.
            """
            self.rolling_periods= rolling_periods
            self.columns = columns

    def __repr__(self) -> str:
         return (
              f"RollingFeature(rolling_periods = {self.rolling_periods},"
              f"columns = {self.columns})"
         )
    def create_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if isinstance(self.rolling_periods,int):
            periods = range(1,self.rolling_periods+1)
        elif isinstance(self.rolling_periods,list):
            periods = self.rolling_periods
        appended_lags = []
        for period in periods: #type:ignore # periods could be Unbound?
            rolling_df = dataframe[self.columns]
            rolling_df.columns=[x+"_roll_avg_"+str(period) for x in self.columns]   
            rolling_df = rolling_df.rolling(window = period).mean()
            appended_lags.append(rolling_df)
        full_list_df = [dataframe] + appended_lags
        return pd.concat(full_list_df, axis=1)