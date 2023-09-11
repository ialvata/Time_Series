from typing import Protocol
import pandas as pd
import numpy as np


class ModelBase(Protocol):
    # we cannot add the properties best_model, or custom_model, since for some models
    # they are simple attributes, for others they are properties.
    best_hyperparameters:dict

    def find_best(self,
        # we only need to include the input which has no default value.
        X_train:pd.DataFrame | pd.Series, y_train: pd.DataFrame | pd.Series,
        # X_val:pd.DataFrame | pd.Series | None = None,
        # y_val: pd.DataFrame | pd.Series | None = None,
        
    )-> None:
        """
        This method should be responsible for the following tasks:
            - Find the optimised hyperparameters values.
            - Fit a model with those optimised values on the *whole* train data.
            - Save the optimised hyperparameters to a instance attribute.
        """
        ...
    def fit_custom_model(self,
        X_train:pd.DataFrame | pd.Series, y_train:pd.DataFrame | pd.Series,
    )-> None:
        """
        This method should be able to also receive other keyword arguments, depending on the
        model being wrapped. They should be gathered with `**kwargs` trick.
        """
        ...
    def forecast(self,
        X_test:pd.DataFrame | pd.Series, use_best_model:bool = False
    )-> float | np.ndarray | pd.DataFrame | pd.Series:...
    