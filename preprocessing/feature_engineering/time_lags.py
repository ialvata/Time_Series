import numpy as np
import pandas as pd

class LagFeature:
    _type = "LagFeature"
    def __init__(self,lags:int|list[int], columns:list[str]) -> None:
            """
            This function will create Time Lag features to mimic an autoregressive model.

            Parameters
            ----------
            `lags:int|list[int]`
                The time lags we want to create in the dataframe.

            `columns:list[str]`
                List of columns names for which we want to build the time lags.

            Note
            ----
            This function will transform a matrix of dim -> n_samples x n_columns
            into a matrix of dim -> (n_samples-lags) x (n_columns*lags).

            TODO:
                Implement case for SeasonLength.WEEK_OF_MONTH.
            """
            self.lags:int|list[int] = lags
            self.columns = columns

    def __repr__(self) -> str:
         return (
              f"LagFeature(lags = {self.lags},"
              f"columns = {self.columns})"
         )
    def create_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if isinstance(self.lags,int):
            lags = range(1,self.lags+1)
        elif isinstance(self.lags,list):
            lags = self.lags
        appended_lags = []
        for lag in lags: #type:ignore
            lag_df= dataframe.shift(lag).drop(columns=self.columns)
            lag_df.columns=[x+"_lag_"+str(lag) for x in lag_df.columns]
            appended_lags.append(lag_df)
        full_list_df = [dataframe] + appended_lags
        return pd.concat(full_list_df, axis=1) 