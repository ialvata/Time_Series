import pandas as pd
from  abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats
from stat_tests.trend_stationarity import adf_test,kpss_test
from preprocessing.transformations_base import Transformation
from preprocessing.pipeline_base import PreprocessPipeline
from preprocessing.pipeline_input_base import PipelineInput


class StationaryInput(PipelineInput):
    def __init__(self, path_to_data: Path, 
                 train_prop: float = 0.8, alpha:float = 0.05):
        """
        alpha
            This parameter determines the test size we're willing 
            to accept a type-I error
        """
        self._dataframe :pd.DataFrame = self.fetch_data(path=path_to_data)
        test_rows = 1 - int(train_prop * self._dataframe.shape[0])
        self.test:pd.DataFrame = self._dataframe.iloc[-test_rows:]
        self._dataframe = self._dataframe.iloc[:test_rows]
        self.alpha = alpha
    
    @property
    def dataframe(self):
        return self._dataframe

class Stationary(PreprocessPipeline):
    """
    This class has several transformations that can help us to stationarize
    the original data.
    """
    def __init__(self, input: StationaryInput):
        PreprocessPipeline.__init__(self,input)

    def trend_stationarity(self):
        pass
