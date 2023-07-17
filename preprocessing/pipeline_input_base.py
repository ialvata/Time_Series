from  abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd

class PipelineInput(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @property
    @abstractmethod
    def dataframe(self):
        pass
    
    def fetch_data(self, path:Path)-> pd.DataFrame:
        # read raw data
        dataframe = pd.read_csv(path)
        return dataframe
    
