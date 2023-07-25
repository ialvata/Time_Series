import pandas as pd
from  abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from preprocessing.transformations_base import Transformation
from preprocessing.preprocess_input_base import PreprocessInput


class Preprocess(ABC):
    def __init__(self, input:PreprocessInput):
        self.input = input

    @property  
    def dataframe(self)->pd.DataFrame:
        return self.input.dataframe
    
    @abstractmethod
    def clean_dataframe(self):
        pass