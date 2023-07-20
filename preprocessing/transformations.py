from preprocessing.stationary import Stationary
from preprocessing.transformations_base import Transformation
import pandas as pd
from scipy.special import inv_boxcox
from scipy.stats import boxcox
from typing import Callable

        

class Difference(Transformation):
    # TODO:
    #   1- Substitute prints with Exceptions.
    #   2- Create decorators to avoid repetition.
    _type = "Difference"
    
    def __init__(self, periods:int, stationary:Stationary | None = None):
        self._parameters = dict(periods=periods)
        self.stationary = stationary
        self.initial_rows = None

    def apply(self, 
              columns: list[str]):
        """ 
        The apply method transforms self.stationary.tranformed_data accordingly, and adds 
        itself to the self.stationary.transformation_pipeline

        Note:
            This method will return a smaller dataframe than the one entered, since the initial
            `periods` rows will be NaN, and dropped.
            This method preserves the column names.      
        """
        if self.stationary is not None:
            periods = self.parameters["periods"]
            columns_indices:list = self.stationary.tranformed_data.columns.get_indexer(
                self.stationary.tranformed_data.columns
            )
            self.initial_rows = self.stationary.tranformed_data.iloc[:periods,columns_indices]
            self.stationary.tranformed_data = self.stationary.tranformed_data[columns].diff(
                periods = periods
            ).dropna()
            self.stationary.add_transformation_to_pipeline(self)
        else:
            print("You should first associate a stationary pipeline to this transformation") 

    def invert(self):
        """
        The dataframe_diff column names must be a subset of Stationary dataframe
        column names.
        """
        if self.stationary is not None:
            if self.stationary.last_transformation_in_pipeline()==self:
                periods = self.parameters["periods"]
                transformed_df = self.stationary.tranformed_data
                if self.initial_rows is not None:
                    concat_df:pd.DataFrame = pd.concat([self.initial_rows, transformed_df])
                    num_rows = concat_df.shape[0]
                    for column in transformed_df.columns:
                        concat_col_values = concat_df[column].values
                        # initial inverted values come from initial dataframe rows.
                        inverted_values = self.initial_rows.values.flatten().tolist()
                        for i in range(periods,num_rows):
                            inverted_value = concat_col_values[i] + inverted_values[i - periods]
                            inverted_values.append(inverted_value)
                        concat_df[column]=inverted_values
                    self.stationary.tranformed_data = concat_df
                else:
                    print("You should first use the apply method to difference the dataframe.")
            else:
                print("The last data transformation was from a different transformation.\
                       Please use the correct transformation")
        else:
            print("You should first associate a stationary pipeline to this transformation.")

class BoxCox(Transformation):

    _type = "BoxCox"

    def __init__(self, 
                 stationary:Stationary | None = None, 
                 lambda_par:float | dict[str,float] | None = None,
                 alpha: float | None = None):
        self._parameters = {
            "lambda":lambda_par,
            "alpha":alpha
        }
        self.stationary = stationary

    def apply(self, 
              columns: list[str]) -> pd.DataFrame | None:
        """
        lambda : {None, scalar}, optional
            If lmbda is not None, do the transformation for that value. 
            If lmbda is None, find the lambda that maximizes the log-likelihood 
                function and return it as the second output argument.
        """
        dict_lambda={}
        transf_dataframe=pd.DataFrame()
        if self.stationary is not None:
            transf_dataframe.index = self.stationary.tranformed_data.index
            for column in columns:
                box_cox_data, lambda_par = boxcox(   
                    self.stationary.input.dataframe[column].values
                )
                transf_dataframe[column]= box_cox_data
                dict_lambda[column]=lambda_par
            self._parameters["lambda"] = dict_lambda
            self._parameters["columns"] = columns
            self.stationary.tranformed_data = transf_dataframe
            self.stationary.add_transformation_to_pipeline(self) 
        else:
            print("You should first associate a stationary pipeline to this transformation")


    def invert(self,
               columns: list[str]|None = None):
        """
        `columns`
            This parameter should be 
        """
        if self.stationary is not None:
            if self.stationary.last_transformation_in_pipeline()==self:
                inverted_dataframe=pd.DataFrame()
                inverted_dataframe.index = self.stationary.tranformed_data.index
                transformed_df = self.stationary.tranformed_data
                if columns is None:
                    columns = self.parameters.get("columns",[])
                for column in columns:
                    lambda_par = self.parameters["lambda"][column]
                    inverted_dataframe[column] = inv_boxcox(
                        transformed_df[column].values,
                        lambda_par
                    )
                self.stationary.tranformed_data = inverted_dataframe
            else:
                print("The last data transformation was from a different transformation.\
                       Please use the correct transformation")
        else:
            print("You should first associate a stationary pipeline to this transformation")


