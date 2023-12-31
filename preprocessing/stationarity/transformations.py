from preprocessing.stationarity.stationary import Stationary
from preprocessing.stationarity.transformations_base import Transformation
import pandas as pd
from scipy.special import inv_boxcox
from scipy.stats import boxcox
from typing import Callable

        

class Difference(Transformation):
    # TODO:
    #   1- Substitute prints with Exceptions.
    #   2- Create decorators to avoid print repetition.
    _type = "Difference"
    
    def __init__(self, periods:int, stationary:Stationary | None = None):
        self._parameters = dict(periods=periods)
        self.stationary = stationary
        self.initial_rows = None

    def transform(self, 
              columns: list[str]):
        """ 
        The transform method transforms `stationary.tranformed_data` accordingly, and adds 
        itself to the `stationary.transformation_pipeline`

        Parameters
        ----------
        columns: list[str]
            A list of strings with the columns names, to which we want to apply this
            transformation.

        Attention
        ---------
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
            Exception("You should first associate a stationary pipeline to this transformation") 

    def invert(self, 
               transformed_df: pd.DataFrame | None = None,
               return_output:bool = False,
               remove_last_in_pipeline: bool = False) -> None | pd.DataFrame:
        """
        The dataframe_diff column names must be a subset of Stationary dataframe
        column names.
        """
        if self.stationary is not None:
            # if self.stationary.current_transformation==self:
                periods = self.parameters["periods"]
                if transformed_df is None:
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
                    if return_output:
                        return concat_df
                    self.stationary.tranformed_data = concat_df
                    # if remove_last_in_pipeline:
                    #     self.stationary.remove_last_in_pipeline()
                else:
                    Exception("You should first use the apply method to difference the dataframe.")
            # else:
            #     Exception("The last data transformation was from a different transformation.\
            #            Please use the correct transformation")
        else:
            Exception("You should first associate a stationary pipeline to this transformation.")

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

    def transform(self, 
              columns: list[str]) -> pd.DataFrame | None:
        """
        The transform method transforms `stationary.tranformed_data` accordingly, and adds 
        itself to the `stationary.transformation_pipeline`

        Parameters
        ----------
        columns: list[str]
            A list of strings with the columns names, to which we want to apply this
            transformation.
        """
        dict_lambda={}
        transf_dataframe=pd.DataFrame()
        if self.stationary is not None:
            transf_dataframe.index = self.stationary.tranformed_data.index
            for column in columns:
                box_cox_data, lambda_par = boxcox(   
                    self.stationary.tranformed_data[column].values
                )
                transf_dataframe[column]= box_cox_data
                dict_lambda[column]=lambda_par
            self._parameters["lambda"] = dict_lambda
            self._parameters["columns"] = columns
            self.stationary.tranformed_data = transf_dataframe
            self.stationary.add_transformation_to_pipeline(self) 
        else:
            Exception("You should first associate a stationary pipeline to this transformation")


    def invert(self,
               transformed_df: pd.DataFrame | None = None,
               columns: list[str]|None = None,
               return_output:bool = False,
               remove_last_in_pipeline: bool = False) -> None | pd.DataFrame:
        """
        Parameters
        ----------
        `return_output`: bool = True
            If `return_output` is True, invert method will return a dataframe after inverting 
            the transformations.
            When `transformed_df` is None, `return_output` will be False.
        `remove_last_in_pipeline`
            Whether to update the transformation pipeline, when running this method. 
        """
        if self.stationary is not None:
            # if self.stationary.current_transformation==self:
                inverted_dataframe=pd.DataFrame()
                if transformed_df is None:
                    return_output = False
                    transformed_df = self.stationary.tranformed_data
                inverted_dataframe.index = transformed_df.index
                if columns is None:
                    columns = self.parameters.get("columns",[])
                for column in columns:
                    lambda_par = self.parameters["lambda"][column]
                    inverted_dataframe[column] = inv_boxcox(
                        transformed_df[column].values,
                        lambda_par
                    )
                if return_output:
                    return inverted_dataframe
                self.stationary.tranformed_data = inverted_dataframe
                # if remove_last_in_pipeline:
                #     self.stationary.remove_last_in_pipeline()
            # else:
            #     Exception("The last data transformation was from a different transformation.\
            #            Please use the correct transformation")
        else:
            Exception("You should first associate a stationary pipeline to this transformation")


