from  abc import ABC, abstractmethod
import pandas as pd

class Transformation(ABC):
    _type:str

    def __init__(self):
        self._parameters = {}
        
    def __repr__(self) -> str:
        if self._parameters is not None:
            return f"{self._type}({self._parameters})"
        else:
            return f"{self._type}({None})"
    @property
    def parameters(self)-> dict:
        return self._parameters
    
    @abstractmethod
    def apply(self):
        pass
    
    @abstractmethod
    def invert(self,transformed_df: pd.DataFrame) -> None:
        pass