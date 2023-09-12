from sklearn.metrics import (
    median_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from typing import Protocol, Union
import numpy as np
import pandas as pd


class MetricBase(Protocol):
    name:str
    def compute(self, 
                predictions: np.ndarray | pd.DataFrame | pd.Series, 
                observations: np.ndarray | pd.DataFrame | pd.Series)-> np.float16:
        ...

class MAE:
    """Median Absolute Error"""
    name:str = "MAE"
    @classmethod
    def compute(cls, predictions: np.ndarray | pd.DataFrame | pd.Series, 
                observations: np.ndarray | pd.DataFrame | pd.Series) -> np.float16:
        return median_absolute_error(y_pred=predictions, y_true=observations) 

class MAPE:
    """Mean Absolute Percentage Error"""
    name:str = "MAPE"
    @classmethod
    def compute(cls, predictions: np.ndarray | pd.DataFrame | pd.Series, 
                observations: np.ndarray | pd.DataFrame | pd.Series) -> float:
        return mean_absolute_percentage_error(y_pred=predictions, y_true=observations)

class MSE:
    """Mean Squared Error"""
    name:str = "MSE"
    @classmethod
    def compute(cls, predictions: np.ndarray | pd.DataFrame | pd.Series, 
                observations: np.ndarray | pd.DataFrame | pd.Series) -> float:
        return mean_squared_error(y_pred=predictions, y_true=observations)

class RMSE:
    """Root Mean Squared Error"""
    name:str = "RMSE"
    @classmethod
    def compute(cls, predictions: np.ndarray | pd.DataFrame | pd.Series, 
                observations: np.ndarray | pd.DataFrame | pd.Series) -> float | np.ndarray:
        return np.sqrt(mean_squared_error(y_pred=predictions, y_true=observations))