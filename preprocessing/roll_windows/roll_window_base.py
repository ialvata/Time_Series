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

class TrainSet:
    def __init__(self, 
                 dataframe:pd.DataFrame,labels_names:list[str], 
                 feature_columns:list[str] | None = None
    ) -> None:
        self.dataframe = dataframe
        self.labels_names = sorted(labels_names)
        self.feature_columns = feature_columns
    
    @property
    def labels(self)-> pd.DataFrame | pd.Series:
        return self.dataframe[self.labels_names]
    
    @property
    def features(self)-> pd.DataFrame | pd.Series:
        # if we do feature engineering, we may want to update self.dataframe after
        # TrainSet initialization. 
        # I'm not sure... Provisional pattern.
        columns_set = set(self.dataframe.columns)
        self.feature_columns = sorted(
            list(columns_set.difference(self.labels_names))
        )
        return self.dataframe[self.labels_names]

class TestSet:
    def __init__(self, 
                 dataframe:pd.DataFrame,labels_names:list[str], 
                 feature_columns:list[str] | None = None
    ) -> None:
        self.dataframe = dataframe
        self.labels_names = sorted(labels_names)
        self.feature_columns = feature_columns
    
    @property
    def labels(self)-> pd.DataFrame | pd.Series:
        return self.dataframe[self.labels_names]
    
    @property
    def features(self)-> pd.DataFrame | pd.Series:
        # if we do feature engineering, we may want to update self.dataframe after
        # TrainSet initialization. 
        # I'm not sure... Provisional pattern.
        columns_set = set(self.dataframe.columns)
        self.feature_columns = sorted(
            list(columns_set.difference(self.labels_names))
        )
        return self.dataframe[self.labels_names]

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

    def create_folds(self) -> Generator[tuple[TrainSet, TestSet],None,None]:
        
        for index in range(self.train_length,self.dataframe.shape[0], self.window):
            yield (
                TrainSet(
                    self.dataframe.iloc[:index],self.labels_names
                ),
                TestSet(
                    self.dataframe.iloc[[index]], self.labels_names
                )
                 
            )

    # def __repr__(self) -> str:
    #     return f"FoldData(feature_shape= {self.feature_train.shape}, \
    #         label_shape= {self.label_test.shape})"