import pandas as pd
from  abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats
from stat_tests.trend_stationarity import adf_test,kpss_test
from preprocessing.stationarity.transformations_base import Transformation
from preprocessing.preprocess_base import Preprocess
from preprocessing.preprocess_input_base import PreprocessInput
from enum import Enum

class TrendStationaryOption(Enum):
    ADF = 1
    KPSS = 2

class StationaryInput(ABC):
    def __init__(self, dataframe: pd.DataFrame, alpha:float = 0.05):
        """
        alpha
            This parameter determines the test size we're willing 
            to accept a type-I error
        """
        # test_rows = int((1-train_prop) * self._dataframe.shape[0])
        # self.test:pd.DataFrame = self._dataframe.iloc[-test_rows:]
        # self._dataframe = self._dataframe.iloc[:-test_rows]
        self.dataframe = dataframe
        self.alpha = alpha
    

class Stationary(ABC):
    """
    This class represents a pipeline/sequence of transformations that create stationary data
    from the original preprocessed data.
    """
    def __init__(self, input: StationaryInput):
        self.transformation_pipeline = []
        # input serves as a backup of original data
        self.input = input
        self.tranformed_data = input.dataframe
    
    def add_transformation_to_pipeline(self,transformation:Transformation):
        self.transformation_pipeline.append(transformation)

    def remove_last_in_pipeline(self):
        self.transformation_pipeline.pop()

    def last_transformation_in_pipeline(self)->Transformation:
        return self.transformation_pipeline[-1]
    
    @abstractmethod
    def stationarize(self):
        pass

    @abstractmethod
    def plot_data(self):
        pass

    def is_trend_stationarity(self,columns:list[str] | None = None,
                              test_option:TrendStationaryOption = TrendStationaryOption.ADF,                                 
                              print_output:bool = False)-> np.bool_|None:
        if self.tranformed_data is not None:
            test_results = []
            self.tranformed_data:pd.DataFrame
            if columns is None:
                columns = list(self.tranformed_data.columns)
            for column in columns:
                match test_option:
                    case TrendStationaryOption.ADF:
                        test_res = adf_test(self.tranformed_data[column], 
                                        print_output = print_output)
                        test_results.append(test_res.p_value<=self.input.alpha)
                    case TrendStationaryOption.KPSS:
                        test_res = kpss_test(self.tranformed_data[column], 
                                        print_output = print_output)
                        test_results.append(test_res.p_value<=self.input.alpha)
                    case _:
                        print("Please choose a correct TrendStationaryOption")
                        break
                
            return np.all(test_results)
