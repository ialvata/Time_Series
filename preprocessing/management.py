"""
This module will contain some utilities related to data management, namely splitting 
into train and test sets, and for the rolling forecasts.
It could also be used in time series cross validation.
"""

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from collections.abc import Generator

class RollingFold:
    def __init__(
        self,
        dataframe:pd.DataFrame,
        window:int = 1,
        train_prop:float = 0.9,
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        train_prop: float in [0,1]
            The proportion of data that should be used for creating the training set.
        window: int
            The `window` gap between test observations 
        """
        self.train_length = int(train_prop*dataframe.shape[0])
        self.dataframe = dataframe
        self.window = window

    def create_folds(self) -> Generator[tuple[pd.DataFrame, pd.DataFrame],None,None]:
        
        for index in range(self.train_length,self.dataframe.shape[0], self.window):
            yield (self.dataframe.iloc[:index], self.dataframe.iloc[[index]])

    # def __repr__(self) -> str:
    #     return f"FoldData(feature_shape= {self.feature_train.shape}, \
    #         label_shape= {self.label_test.shape})"
