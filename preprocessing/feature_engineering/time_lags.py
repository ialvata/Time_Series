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

                Attention:
                ------
                These columns must be part of the initial dataframe, pre-transformed by
                the FeatureEngineering pipeline.

            Note
            ----
            This function will transform a matrix of dim -> n_samples x n_columns
            into a matrix of dim -> (n_samples-lags) x (n_columns*lags).

            TODO:
                Implement case for SeasonLength.WEEK_OF_MONTH.
            """
            self.columns = columns
            if isinstance(lags,int):
                self.lags = range(1,lags+1)
            elif isinstance(lags,list):
                self.lags = lags
            else:
                 raise Exception("The lags variable is of the wrong data type!")
            self.num_rows_train = max(self.lags)

    def __repr__(self) -> str:
         return (
              f"LagFeature(lags = {self.lags},"
              f"columns = {self.columns})"
         )
    def create_features(self, dataframe: pd.DataFrame, 
                        auxiliary_df: pd.DataFrame | None = None) -> pd.DataFrame:
        if auxiliary_df is not None:
            original_columns = [list(dataframe.columns).index(col) for col in self.columns]
            if max(original_columns)>len(auxiliary_df.columns)-1:
                raise Exception("Chosen columns are not part of the original dataframe")
            dataframe = pd.concat(
               [auxiliary_df.iloc[
                   -self.num_rows_train:,
                   # we want to retrieve only the feature columns for which we want to create
                   # the rolling features.
                   original_columns
                ], dataframe],
                axis=0)
        appended_lags = []
        for lag in self.lags: #type:ignore
            lag_df= dataframe.shift(lag)[self.columns]
            lag_df.columns=[x+"_lag_"+str(lag) for x in lag_df.columns]
            appended_lags.append(lag_df)
        full_list_df = [dataframe] + appended_lags
        return pd.concat(full_list_df, axis=1).dropna() 