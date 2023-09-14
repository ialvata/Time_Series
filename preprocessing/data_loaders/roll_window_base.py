"""
This module will contain some utilities related to data management, namely splitting 
into train and test sets, and for the rolling forecasts.
It could also be used in time series cross validation.
"""

import pandas as pd
from collections.abc import Generator
from  abc import ABC, abstractmethod
from preprocessing.data_spliting.set_splitting import Split

class RollWindow(ABC):
    @abstractmethod
    def __init__(self, 
                 data_split: Split,
                 label_columns: list[str] | None = None):
        pass

class ClassicalWindow(RollWindow):
    def __init__(
        self,
        dataframe:pd.DataFrame,
        labels_names:list[str],
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
        self.labels_names = labels_names

    def create_folds(self) -> Generator[tuple[pd.DataFrame, pd.DataFrame],None,None]:
        
        for index in range(self.train_length,self.dataframe.shape[0], self.window):
            yield (
                self.dataframe.iloc[:index],self.dataframe.iloc[[index]]
            )

    # def __repr__(self) -> str:
    #     return f"FoldData(feature_shape= {self.feature_train.shape}, \
    #         label_shape= {self.label_test.shape})"