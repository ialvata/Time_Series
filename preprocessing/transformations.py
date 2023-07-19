from preprocessing.pipeline_base import PreprocessPipeline
from transformations_base import Transformation
import pandas as pd


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


