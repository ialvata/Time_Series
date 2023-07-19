from  abc import ABC, abstractmethod

class Transformation(ABC):
    _type:str

    def __init__(self):
        self._parameters = {}

    @property
    def parameters(self)-> dict:
        return self._parameters
    
    @abstractmethod
    def apply(self):
        pass
    
    @abstractmethod
    def invert(self):
        pass