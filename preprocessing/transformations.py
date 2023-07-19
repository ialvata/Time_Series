from preprocessing.stationary import Stationary
from preprocessing.transformations_base import Transformation
import pandas as pd
from scipy.special import boxcox, inv_boxcox
from typing import Callable

        

class Difference(Transformation):
    _type = "Difference"
    
    def __init__(self, periods:int, stationary:Stationary | None = None):
        self._parameters = dict(periods=periods)
        self.stationary = stationary

    def apply(self, 
              columns: list[str]) -> pd.DataFrame | None:
        """ 
        This method will return a smaller dataframe than the one entered, since the initial
        `periods` rows will be NaN, and dropped.
        This method preserves the column names.      
        """
        if self.stationary is not None:
            return self.stationary.input.dataframe[columns].diff(
                periods=self.parameters["periods"]
            ).dropna()
        else:
            print("You should first associate a stationary pipeline to this transformation") 

    def invert(self, dataframe_diff:pd.DataFrame, 
               initial_rows:pd.DataFrame | None = None):
        """
        The dataframe_diff column names must be a subset of Stationary dataframe
        column names.
        """
        if self.stationary is not None:
            periods = self.parameters["periods"]
            columns_indices:list = self.stationary.input.dataframe.columns.get_indexer(
                dataframe_diff.columns
            )
            if initial_rows is None:
                initial_rows = self.stationary.input.dataframe.iloc[:periods,columns_indices]
            concat_df:pd.DataFrame = pd.concat([initial_rows, dataframe_diff])
            num_rows = concat_df.shape[0]
            for column in dataframe_diff.columns:
                concat_col_values = concat_df[column].values
                # initial inverted values come from initial dataframe rows.
                inverted_values = initial_rows.values.flatten().tolist()
                for i in range(periods,num_rows):
                    inverted_value = concat_col_values[i] + inverted_values[i - periods]
                    inverted_values.append(inverted_value)
                concat_df[column]=inverted_values
            return concat_df
        else:
            print("You should first associate a stationary pipeline to this transformation")






# class BoxCoxTrans(Transformation):
#     _type = "BoxCox"
#     def __init__(self, 
#                  stationary:Stationary | None = None, 
#                  lambda_par:float | None = None,
#                  alpha: float | None = None):
#         self._parameters = {
#             "lambda":lambda_par,
#             "alpha":alpha
#         }
#         self.stationary = stationary

        
#     def apply(self, 
#               columns: list[str]) -> pd.DataFrame | None:
#         """
#         lambda : {None, scalar}, optional
#             If lmbda is not None, do the transformation for that value. 
#             If lmbda is None, find the lambda that maximizes the log-likelihood 
#                 function and return it as the second output argument.
#         """
#         dict_lambda={}
#         transf_dataframe=pd.DataFrame()
#         if self.stationary is not None:
#             for column in columns:
#                 box_cox_data, lambda_par = boxcox(
#                     self.stationary.input.dataframe[column].values,
#                     lambda
#                 )
#                 transf_dataframe[column]= box_cox_data
#                 dict_lambda[column]=lambda_par
#             self._parameters["lambda"] = dict_lambda
#             self._parameters["columns"] = columns
#             return transf_dataframe
#         else:
#             print("You should first associate a stationary pipeline to this transformation")


#     def invert(self,
#                transf_dataframe:pd.DataFrame,
#                columns: list[str]|None = None) -> pd.DataFrame:
#         """
#         `columns`
#             This parameter should be 
#         """
#         inverted_dataframe=pd.DataFrame()
#         if columns is None:
#             columns = self.parameters.get("columns",[])
#         for column in columns:
#             lambda_par = self.parameters["lambda"][column]
#             inverted_dataframe[column] = inv_boxcox(
#                 transf_dataframe[column].values,
#                 lambda_par
#             )
#         return inverted_dataframe


