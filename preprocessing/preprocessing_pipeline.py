import pandas as pd
from  abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats

class Transformation(ABC):
    _type:str
    @property
    @abstractmethod
    def parameters(self):
        pass
    @abstractmethod
    def apply(self):
        pass
    @abstractmethod
    def invert(self):
        pass
class PreprocessPipeline(ABC):
    def __init__(self, path_to_data: Path, train_prop: float):
        if path_to_data:
            self.dataframe :pd.DataFrame = self.fetch_data(path=path_to_data)
        self.pipeline = []
        test_rows = 1 - int(train_prop * self.dataframe.shape[0])
        self.test:pd.DataFrame = self.dataframe.iloc[-test_rows:]
        self.dataframe = self.dataframe.iloc[:test_rows]
    
    def add_transformation_to_pipeline(self,transformation:Transformation):
        self.pipeline.append(transformation)

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

class Difference(Transformation):
    _type = "Difference"
    
    def __init__(self, pipeline:PreprocessPipeline, periods:int):
        self._parameters = dict(periods=periods)
        self.pipeline = pipeline

    @property
    def parameters(self)-> dict:
        return self._parameters

    def apply(self, 
              columns: list[str]) -> pd.DataFrame:
        """ 
        This method will return a smaller dataframe than the one entered, since the initial
        `periods` rows will be NaN, and dropped.
        This method preserves the column names.      
        """
        return self.pipeline.dataframe[columns].diff(
            periods=self.parameters["periods"]
        ).dropna()
    
    def invert(self,
               dataframe_diff:pd.DataFrame):
        """
        dataframe_diff column names must be a subset of self.pipeline.dataframe
        column names
        
        """
        periods = self.parameters["periods"]
        columns_indices:list = self.pipeline.dataframe.columns.get_indexer(
            dataframe_diff.columns
        )
        initial_rows = self.pipeline.dataframe.iloc[:periods,columns_indices]
        concat_df:pd.DataFrame = pd.concat([initial_rows, dataframe_diff])
        return concat_df.cumsum()

class BoxCoxTrans(Transformation):
    _type = "BoxCox"
    def __init__(self, pipeline:PreprocessPipeline, 
                 lambda:float | None = None,
                 alpha: float | None = None):
        self._parameters = {
            "lambda":lambda,
            "alpha":alpha
        }
        self.pipeline = pipeline

    @property
    def parameters(self)-> dict:
        return self._parameters
    
    def apply(self, 
              columns: list[str]) -> pd.DataFrame:
        """
        lambda : {None, scalar}, optional
            If lmbda is not None, do the transformation for that value. 
            If lmbda is None, find the lambda that maximizes the log-likelihood 
                function and return it as the second output argument.
        """
        dict_lambda={}
        transf_dataframe=pd.DataFrame()
        for column in columns:
            box_cox_data, lambda_par = stats.boxcox(
                self.pipeline.dataframe[column].values,
                lmbda = lambda
            )
            transf_dataframe[column]= box_cox_data
            dict_lambda[column]=lambda_par
        self._parameters["lambda"] = dict_lambda
        self._parameters["columns"] = columns
        return transf_dataframe

    def invert(self,
               transf_dataframe:pd.DataFrame,
               columns: list[str]|None = None) -> pd.DataFrame:
        """
        `columns`
            This parameter should be 
        """
        inverted_dataframe=pd.DataFrame()
        if columns is None:
            columns = self.parameters.get("columns",[])
        for column in columns:
            lambda_par = self.parameters["lambda"][column]
            inverted_dataframe[column] = stats.inv_boxcox(
                transf_dataframe[column].values,
                lmbda = lambda_par
            )
        return inverted_dataframe
        
class Stationarize(PreprocessPipeline):
    """
    This class has several transformations that can help us to stationarize
    the original data.
    """
        


