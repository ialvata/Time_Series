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

    # def create_folds(self) -> Generator[tuple[pd.DataFrame, pd.DataFrame],None,None]:
        
    #     for index in range(self.train_length,self.dataframe.shape[0], self.window):
    #         yield (self.dataframe.iloc[:index], self.dataframe.iloc[[index]])

    # def __repr__(self) -> str:
    #     return f"FoldData(feature_shape= {self.feature_train.shape}, \
    #         label_shape= {self.label_test.shape})"
