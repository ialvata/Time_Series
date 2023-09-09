from typing import Protocol
import pandas as pd
import numpy as np


class ModelBase(Protocol):
    def find_best(self,
        X_train:pd.DataFrame | pd.Series, y_train: pd.DataFrame | pd.Series,
        X_val:pd.DataFrame | pd.Series | None = None,
        y_val: pd.DataFrame | pd.Series | None = None,
        # num_obs_val: int = 10, # This needs to be changed to a proportion
        # show_progress_bar: bool = True
    )-> None:...
    def fit_custom_model(self,
        X_df:pd.DataFrame | pd.Series, y_df:pd.DataFrame | pd.Series
    )-> None:...
    def forecast(self,
        X_test:pd.DataFrame | pd.Series, use_best_model:bool = False
    )-> np.ndarray:...
    