import pandas as pd
from  abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats
from stat_tests.trend_stationarity import adf_test,kpss_test
from transformations_base import Transformation
from pipeline_base import PreprocessPipeline
from pipeline_input_base import PipelineInput


class StationaryInput(PipelineInput):
    def __init__(self, path_to_data: Path, 
                 train_prop: float, alpha:float):
        """
        alpha
            This parameter determines the test size we're willing 
            to accept a type-I error
        """
        self.dataframe :pd.DataFrame = self.fetch_data(path=path_to_data)
        test_rows = 1 - int(train_prop * self.dataframe.shape[0])
        self.test:pd.DataFrame = self.dataframe.iloc[-test_rows:]
        self.dataframe = self.dataframe.iloc[:test_rows]

class Stationary(PreprocessPipeline):
    """
    This class has several transformations that can help us to stationarize
    the original data.
    """
    def __init__(self, input: StationaryInput):
        PreprocessPipeline.__init__(self,input)

    def trend_stationarity(self):
        pass
