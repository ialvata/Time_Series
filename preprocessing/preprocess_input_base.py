from  abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd

class PreprocessInput(ABC):
    
    def __init__(self,path_to_data):
        self._dataframe :pd.DataFrame = self.fetch_data(path=path_to_data)

    @property
    def dataframe(self)-> pd.DataFrame:
        return self._dataframe
    
    def fetch_data(self, path:Path)-> pd.DataFrame:
        # read raw data
        dataframe = pd.read_csv(path)
        return dataframe
    
