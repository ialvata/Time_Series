from sklearn.metrics import (
    median_absolute_error as mae,
    mean_absolute_percentage_error as mape,
    mean_squared_error as mse,
)
from math import sqrt
import numpy as np

def rmse(values:np.ndarray)-> float:
    return np.sqrt(mse(values))