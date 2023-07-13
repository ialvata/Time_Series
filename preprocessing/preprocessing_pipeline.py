import pandas as pd
from  abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class PreprocessingPipeline(ABC):
    def __init__(self, path_to_data: Path | None = None):
        if path_to_data:
            self.dataframe :pd.DataFrame = self.fetch_data(path=path_to_data)

    def add_lags(self, df:pd.DataFrame, n_lags:int|list[int], drop_cols:list)-> pd.DataFrame:
        """
        Each row of df must be a separate time point, which will be transformed
        into a lag. This function will transform a matrix of dim -> n_samples x n_columns
        into a matrix of dim -> (n_samples-n_lags) x (n_columns*n_lags)
        """
        if isinstance(n_lags,int):
            lags = range(1,n_lags+1)
        elif isinstance(n_lags,list):
            lags = n_lags
        appended_lags = []
        for lag in lags: 
            lag_df= df.shift(lag).drop(columns=drop_cols)
            lag_df.columns=[x+"_lag_"+str(lag) for x in lag_df.columns]
            appended_lags.append(lag_df)
        return (
            pd.concat(appended_lags, axis=1)\
                .dropna()\
                # droping the empty cells caused by shifting
                .reset_index(drop=True)\
                # resetting index so that .loc[0] gives 1st row
        )

    def fetch_data(self, path:Path)-> pd.DataFrame:
        # read raw data
        dataframe = pd.read_csv(path)
        return dataframe
