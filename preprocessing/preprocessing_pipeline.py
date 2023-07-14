import pandas as pd
from  abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats

class PreprocessPipeline(ABC):
    def __init__(self, path_to_data: Path, train_prop: float):
        if path_to_data:
            self.dataframe :pd.DataFrame = self.fetch_data(path=path_to_data)
        test_rows = 1 - int(train_prop * self.dataframe.shape[0])
        self.test:pd.DataFrame = self.dataframe.iloc[-test_rows:]
        self.dataframe = self.dataframe.iloc[:test_rows]

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


class Stationarize(PreprocessPipeline):
    """
    This class has several transformations that can help us to stationarize
    the original data.
    """
    def apply_diff(self,columns: list[str], periods:int) -> pd.DataFrame:
        """ 
        This method will return a smaller dataframe than the one entered, since the initial
        `periods` rows will be NaN, and dropped.       
        """
        return self.dataframe[columns].diff(periods=periods).dropna()
    
    def invert_diff(self,df_diff:pd.DataFrame, periods:int):
        columns_indices = self.dataframe.columns.get_indexer(df_diff.columns)
        initial_rows = self.dataframe.iloc[:periods,columns_indices]
        concat_df:pd.DataFrame = pd.concat([initial_rows, df_diff])
        return concat_df.cumsum()
        
    
    def apply_box_cox(self, columns: list[str], 
                      lambda:float | None = None,
                      alpha: float | None = None) -> tuple[pd.DataFrame,dict]:
        """
        lambda : {None, scalar}, optional
            If lmbda is not None, do the transformation for that value. 
            If lmbda is None, find the lambda that maximizes the log-likelihood 
                function and return it as the second output argument.
        """
        dict_lambda={}
        df_box_cox=pd.DataFrame()
        for column in columns:
            box_cox_data, lambda_par = stats.boxcox(self.dataframe[column].values,
                                                    lmbda = lambda)
            df_box_cox[column]= box_cox_data
            dict_lambda[column]=lambda_par
        return df_box_cox, dict_lambda

    def invert_box_cox(self):
        stats.inv_boxcox(data_series, lambda)