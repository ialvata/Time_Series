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

    def plot_heatmap(self):
        """
        This method plots a correlation heatmap using all the features in the dataframe.
        """
        plt.figure(figsize=(8, 8))
        plt.matshow(self.dataframe.corr(),fignum=0)
        plt.xticks(range(self.dataframe.shape[1]), self.dataframe.columns, 
                   fontsize=10, rotation=90)
        plt.gca().xaxis.tick_bottom()
        plt.yticks(range(self.dataframe.shape[1]), self.dataframe.columns, fontsize=10)

        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=10)
        plt.title("Feature Correlation Heatmap", fontsize=14)
        plt.tight_layout()
        plt.show()